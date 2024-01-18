from typing import TYPE_CHECKING

import torch
from torch.utils.hooks import RemovableHandle

from .histogram_calibrator import HistogramCalibrator

if TYPE_CHECKING:
    from ..nn.fake_quantizer import FakeQuantizer


class PercentileCalibrator(HistogramCalibrator):
    """Percentile Calibrator Class

    Attributes:
        quantizer (FakeQuantizer): The `FakeQuantizer` module to be calibrated.
        percentile (float): The desired percentile value, ranging from 0 to 100.

    """

    def __init__(self, quantizer: "FakeQuantizer", percentile: float):
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

    def update(self) -> None:
        # update step_size using "percentile"
        super().update()
        assert isinstance(self.hook_handler, RemovableHandle)

        # cumsum_cuda_kernel does not have a deterministic implementation
        _deterministic_enable_status = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False, warn_only=True)

        for chn, _ in enumerate(self.histc_bins):
            total = self.histogram[chn].data.sum()
            cdf = torch.cumsum(self.histogram[chn].data / total, 0)
            idx = torch.searchsorted(cdf, self.percentile / 100)
            per_max = self.bin_edges[chn].data[idx]
            self.quantizer.step_size.data[chn] = (
                (per_max / self.quantizer.maxabs_bound).detach().to(self.quantizer.step_size.device)
            )

        # allocate deterministic algorithms to original state
        torch.use_deterministic_algorithms(_deterministic_enable_status, warn_only=True)

        self.clear_attribution()
