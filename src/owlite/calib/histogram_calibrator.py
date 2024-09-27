from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.hooks import RemovableHandle

from ..core.constants import OWLITE_CALIBRATOR_HISTOGRAM_SIZE
from ..core.logger import log
from .calibrator import Calibrator

if TYPE_CHECKING:
    from ..nn import FakeQuantizer


class HistogramCalibrator(Calibrator, ABC):
    """Histogram Calibrator Class.

    Attributes:
        histograms(`list[torch.Tensor]`): list of histogram counts. Each element defaults to [0, ..., 0].
        bin_edges(`list[torch.Tensor]`): histogram edges. Each element defaults to [0, ..., 0].
    """

    def __init__(self, quantizer: "FakeQuantizer"):
        """Initialize for histogram calibrator."""
        super().__init__(quantizer)
        self.histograms: list[torch.Tensor] = []
        self.bin_edges: list[torch.Tensor] = []

    def prepare(self) -> RemovableHandle:
        """Prepare forward hook function."""

        def histogram_forward_hook_func(module: "FakeQuantizer", inputs: tuple[Any, ...], output: Any) -> Any | None:
            """Forward hook function to get histogram value."""
            assert isinstance(module.calibrator, HistogramCalibrator)
            calibrator = module.calibrator
            assert self.check_calib_ready()

            if any(len(hist_attr) == 0 for hist_attr in (self.histograms, self.bin_edges)):
                log.error(f"`histogram`: {calibrator.histograms}\n`bin_edge`: {calibrator.bin_edges}")
                raise ValueError("During calibration, calibration attributions are not initialized")

            _input = inputs[0].clone()

            if module.symmetric and module.unsigned and inputs[0].min() < 0:
                log.warning(
                    f"An unsigned fake quantizer (id: '{module.id}') called with a tensor containing a negative value. "
                    "It will be automatically converted to a signed fake quantizer",
                    stacklevel=2,
                )  # UX
                module.invert_signedness()

            new_input = []
            if module.per_channel and (channel := module.channel) is not None:
                for chn in range(channel.size):
                    _input_chn = torch.select(_input, channel.axis, chn)
                    new_input.append(_input_chn)
            else:
                new_input.append(_input)

            # _histc_cuda does not have a deterministic implementation
            _deterministic_enable_status = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(False, warn_only=True)
            if module.symmetric:
                _accumulate_input_to_abs_histogram(calibrator, new_input)
            else:
                _accumulate_input_to_histogram(calibrator, new_input)

            # allocate deterministic algorithms to original state
            torch.use_deterministic_algorithms(_deterministic_enable_status, warn_only=True)

            return output

        # ~define forward hook function

        # set histogram, bin_edges attr and register forward hook
        if (channel := self.quantizer.channel) is not None:
            channel_size = channel.size
        else:
            channel_size = 1
        device = self.quantizer.step_size.device

        if any(len(hist_attr) != 0 for hist_attr in (self.histograms, self.bin_edges)):
            log.error(
                "The histogram attributions are already set before the calibration is prepared.\n"
                f"`histogram`: {self.histograms}\n`bin_edges`: {self.bin_edges}"
            )
            raise ValueError("The histogram attributions are already set before the calibration is prepared")

        self.histograms = [torch.zeros(OWLITE_CALIBRATOR_HISTOGRAM_SIZE).to(device) for _ in range(channel_size)]
        self.bin_edges = [torch.zeros(OWLITE_CALIBRATOR_HISTOGRAM_SIZE + 1).to(device) for _ in range(channel_size)]

        self.hook_handler = self.quantizer.register_forward_hook(histogram_forward_hook_func)
        return self.hook_handler

    @abstractmethod
    def update(self) -> None:
        assert self.check_calib_ready()
        if any(len(hist_attr) == 0 for hist_attr in (self.histograms, self.bin_edges)):
            log.error(f"`histogram`: {self.histograms}\n `bin_edge`: {self.bin_edges}")
            raise ValueError("During calibration, calibration attributions are not initialized")

    def clear(self) -> None:
        """Clear attributes of histogram(`histogram`, `bin_edges`) and registered forward_hook."""
        assert isinstance(self.hook_handler, RemovableHandle)
        self.histograms.clear()
        self.bin_edges.clear()

        # remove registered forward_hook
        self.hook_handler.remove()
        self.hook_handler = None


def _accumulate_input_to_abs_histogram(calibrator: HistogramCalibrator, inputs: list[torch.Tensor]) -> None:
    for i, val in enumerate(inputs):
        local_max = val.abs().max().item()
        histogram_tensor = calibrator.histograms[i]
        bin_edge = calibrator.bin_edges[i]
        histc_bin = len(histogram_tensor)
        if histogram_tensor.sum() == 0 and bin_edge.sum() == 0:
            histogram_tensor.data = torch.histc(val.abs(), bins=histc_bin, min=0, max=local_max).to(
                histogram_tensor.device
            )
            bin_edge.data = torch.linspace(0, local_max, histc_bin + 1).to(bin_edge.device)
        elif calibrator.quantizer.per_channel:
            break
        else:
            if local_max > bin_edge[-1]:
                interval = (bin_edge[1] - bin_edge[0]).item()
                histc_bin = int(round(local_max / interval))
                bin_edge.data = torch.arange(0, local_max + interval, interval, device=bin_edge.device)
            local_hist = torch.histc(val.abs(), histc_bin, 0, float(bin_edge[-1])).to(histogram_tensor.device)
            local_hist[: histogram_tensor.numel()] += histogram_tensor.data
            histogram_tensor.data = local_hist


def _accumulate_input_to_histogram(calibrator: HistogramCalibrator, inputs: list[torch.Tensor]) -> None:
    for i, val in enumerate(inputs):
        local_max = val.max().item()
        local_min = val.min().item()
        histogram_tensor = calibrator.histograms[i]
        bin_edge = calibrator.bin_edges[i]
        histc_bin = len(histogram_tensor)
        if histogram_tensor.sum() == 0 and bin_edge.sum() == 0:
            histogram_tensor.data = torch.histc(val, bins=histc_bin, min=local_min, max=local_max).to(
                histogram_tensor.device
            )
            bin_edge.data = torch.linspace(local_min, local_max, histc_bin + 1).to(bin_edge.device)
        elif calibrator.quantizer.per_channel:
            break
        else:
            min_index = 0
            if local_max > bin_edge[-1] or local_min < bin_edge[0]:
                interval = (bin_edge[1] - bin_edge[0]).item()
                min_index = int(max(torch.ceil((bin_edge[0] - local_min) / interval).item(), 0))
                new_min = (bin_edge[0] - min_index * interval).item()
                new_max = max(local_max, bin_edge[-1].item())
                bin_edge.data = torch.arange(new_min, new_max + interval, interval, device=bin_edge.device)
                histc_bin = len(bin_edge) - 1
            local_hist = torch.histc(val, histc_bin, float(bin_edge[0]), float(bin_edge[-1])).to(
                histogram_tensor.device
            )
            local_hist[min_index : min_index + histogram_tensor.numel()] += histogram_tensor.data
            histogram_tensor.data = local_hist
