from collections import Counter
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..enums import TargetDType
from .histogram_calibrator import HistogramCalibrator

if TYPE_CHECKING:
    from ..nn import FakeINTQuantizer


# pylint: disable=too-many-locals
class EntropyCalibrator(HistogramCalibrator):
    r"""EntropyCalibrator class.

    The EntropyCalibrator compares the distribution of original data and
    the distribution of quantized data using KL divergence.
    When the original data $$ X $$ and the quantized data $$ X_{quant} $$ is given,
    the $$step\\_size$$ is calculated as follow:

    $$
    step\\_size = \underset {step\\_size}{\\operatorname{argmax}} \\,
    KL \\left( X || X_{quant} \right)
    $$

    This approach minimizes the divergence between two distributions.
    """

    searching_ratio = 0.75
    stride = 24

    def __init__(self, quantizer: "FakeINTQuantizer"):
        if quantizer.target_dtype == (TargetDType.fp8_e4m3):
            raise NotImplementedError("EntropyCalibrator for fp8_e4m3 is not implemented")
        super().__init__(quantizer)

    def update(self) -> None:
        """Update step_size using "`entropy`"."""
        super().update()

        max_values, min_values = torch.empty_like(self.quantizer.step_size), torch.empty_like(self.quantizer.step_size)
        for chn, (histogram, bin_edge) in enumerate(zip(self.histograms, self.bin_edges)):
            bins = histogram.detach().cpu().numpy().astype(np.int32)
            valid = bins != 0
            stop = len(bins)

            min_divergence = np.inf
            if self.quantizer.symmetric:
                bins[0] = bins[1]
                valid[0] = valid[1]
                start_idx = int(stop * (1 - self.searching_ratio))
                for max_idx in range(stop, start_idx, -self.stride):
                    if not valid[max_idx - 1]:
                        continue
                    divergence = self._symmetric_quantization_divergence(bins, valid, max_idx)
                    if divergence <= min_divergence:
                        min_divergence = divergence
                        max_values[chn] = (bin_edge[max_idx] + bin_edge[max_idx - 1]) / 2
            else:
                distribution = ((bin_edge[:-1] + bin_edge[1:]) * 0.5).detach().cpu().numpy()
                valid_bins = bins[valid]
                valid_distribution = distribution[valid]
                left_bound = range(int(stop * 0.5 * (1 - self.searching_ratio)), 0, -ceil(self.stride / 2))
                right_bound = range(int(stop * 0.5 * (1 + self.searching_ratio)), stop - 1, ceil(self.stride))
                for min_idx, max_idx in zip(left_bound, right_bound):
                    if min_idx > max_idx:
                        continue
                    max_value = max(distribution[max_idx], 0)
                    min_value = min(distribution[min_idx], 0)
                    divergence = self._asymmetric_quantization_divergence(
                        valid_distribution, valid_bins, max_value, min_value
                    )
                    if divergence <= min_divergence:
                        min_divergence = divergence
                        min_values[chn] = float(min_value)
                        max_values[chn] = float(max_value)

        self.update_fake_quantizer_param_with_max_min(max_values, min_values)
        self.clear()

    def _symmetric_quantization_divergence(self, bins: np.ndarray, valid: np.ndarray, max_idx: int) -> float:
        nbins = int(self.quantizer.maxabs_bound + 1)
        valid_bins = bins[:max_idx][valid[:max_idx]]
        valid_digitized_space = (np.digitize(range(max_idx), np.linspace(0, max_idx, num=nbins + 1)) - 1)[
            valid[:max_idx]
        ]

        new_density_counts = np.zeros(nbins, dtype=np.float32)
        for idx, digitized in enumerate(valid_digitized_space):
            new_density_counts[digitized] += valid_bins[idx]
        counter = Counter(valid_digitized_space)
        for key, val in counter.items():
            new_density_counts[key] = new_density_counts[key] / val

        new_density = new_density_counts[valid_digitized_space].astype(np.float32)
        new_density /= new_density.sum()

        reference_density = valid_bins.astype(np.float32)
        reference_density[-1] += np.sum(bins[max_idx:])
        reference_density /= reference_density.sum()

        divergence = np.sum(reference_density * np.log(reference_density / new_density))
        return divergence

    def _asymmetric_quantization_divergence(
        self, distribution: np.ndarray, histogram: np.ndarray, max_value: float, min_value: float
    ) -> float:
        nbins = int(self.quantizer.quant_max - self.quantizer.quant_min + 1)
        scale = np.float32(max_value - min_value) / (self.quantizer.quant_max - self.quantizer.quant_min)
        zero_point = -(min_value / scale).round() + self.quantizer.quant_min
        if zero_point < self.quantizer.quant_min or zero_point > self.quantizer.quant_max:
            raise ValueError("The quantization range of zero_point has been exceeded")

        valid_digitized_space = (
            ((distribution / scale).round() + zero_point)
            .clip(self.quantizer.quant_min, self.quantizer.quant_max)
            .astype(np.int32)
        )

        new_density_counts = np.zeros(nbins, dtype=np.float64)
        for idx, digitized in enumerate(valid_digitized_space):
            new_density_counts[digitized] += histogram[idx]
        counter = Counter(valid_digitized_space)
        for key, val in counter.items():
            new_density_counts[key] = new_density_counts[key] / val
        new_density = new_density_counts[valid_digitized_space].astype(np.float32)
        new_density /= new_density.sum()

        reference_density = histogram.astype(np.float32)
        reference_density /= reference_density.sum()

        divergence = np.sum(reference_density * np.log(reference_density / new_density))
        return divergence
