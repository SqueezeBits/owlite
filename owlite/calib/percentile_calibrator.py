"""Percentile calibrator"""
import torch

from ._histogram_calibrator import _HistogramCalibrator


class PercentileCalibrator(_HistogramCalibrator):
    """Percentile Calibrator Class

    Attributes:
        quantizer (FakeQuantizer): The `FakeQuantizer` module to be calibrated.
        percentile (float): The desired percentile value, ranging from 0 to 100.

    """

    def __init__(self, quantizer, percentile: float):
        """Initializes the percentile calibrator.

        Args:
            quantizer (FakeQuantizer): The `FakeQuantizer` module to be calibrated.
            percentile(float): The desired percentile value, ranging from 0 to 100.
        Raises:
            ValueError: If the percentile is outside the valid range [0, 100].
        """
        super().__init__(quantizer)
        if percentile < 0 or percentile > 100:
            raise ValueError("percentile must be in range [0,100]")
        self.percentile = percentile

    def update(self):
        # update step_size using "percentile"

        # cumsum_cuda_kernel does not have a deterministic implementation
        _deterministic_enable = torch.are_deterministic_algorithms_enabled()
        if _deterministic_enable:
            torch.use_deterministic_algorithms(False)

        for chn, _ in enumerate(self.quantizer.histc_bins):
            total = self.quantizer.histogram[chn].data.sum()
            cdf = torch.cumsum(self.quantizer.histogram[chn].data / total, 0)
            idx = torch.searchsorted(cdf, self.percentile / 100)
            per_max = self.quantizer.bin_edges[chn].data[idx]
            self.quantizer.step_size.data[chn] = (
                (per_max / self.quantizer.maxabs_bound)
                .detach()
                .to(self.quantizer.step_size.device)
            )

        # allocate to original state
        if _deterministic_enable:
            torch.use_deterministic_algorithms(True)

        # delete "histogram" attritbution from quantizer
        for key in self.set_attr_list:
            delattr(self.quantizer, key)
        # remove registered forward_hook.item())
        self.hook_handler.remove()
