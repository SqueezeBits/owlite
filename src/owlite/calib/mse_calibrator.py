from typing import TYPE_CHECKING

import numpy as np
import torch

from ..enums import TargetDType
from .histogram_calibrator import HistogramCalibrator

if TYPE_CHECKING:
    from ..nn import FakeQuantizer


class MSECalibrator(HistogramCalibrator):
    r"""MSE Calibrator Class.

    The MSECalibrator solves the $$ step\_size $$ that minimizeds the mean squared error (MSE)
    between the original data and its quantized representation.
    When the original data $$ X $$ and the quantized representation $$ X_{quant} $$ is given,
    the optimal $$ step\_size $$ is calculated as follow:

    $$
    step\_size = \underset {step\_size}{\operatorname{argmax}} \,
    {\left\| X - X_{quant} \right\|}^2_2
    $$

    This approach minimizes the mean squared error between two data.
    """

    searching_ratio = 0.5
    stride = 12

    def __init__(self, quantizer: "FakeQuantizer"):
        if quantizer.target_dtype == (TargetDType.fp8_e4m3):
            raise NotImplementedError("MSECalibrator for fp8_e4m3 is not implemented")
        super().__init__(quantizer)

    def update(self) -> None:
        """Update step_size using "`mse`"."""
        super().update()
        max_values, min_values = torch.empty_like(self.quantizer.step_size), torch.empty_like(self.quantizer.step_size)
        for chn, _ in enumerate(self.histograms):
            bins = self.histograms[chn].detach().cpu().numpy().astype(np.int32)
            valid = bins != 0
            valid_bins = bins[valid]
            min_mse = np.inf
            stop = len(bins)

            if self.quantizer.symmetric:
                start_idx = int(stop * (1 - self.searching_ratio))
                for max_idx in range(start_idx, stop + 1, self.stride):
                    mse = self._symmetric_quantization_normalized_mse(valid_bins, valid, max_idx)
                    if mse <= min_mse:
                        min_mse = mse
                        max_values[chn] = (self.bin_edges[chn][max_idx] + self.bin_edges[chn][max_idx - 1]) / 2
            else:
                distribution = ((self.bin_edges[chn][:-1] + self.bin_edges[chn][1:]) * 0.5).detach().cpu().numpy()
                valid_distribution = distribution[valid]
                left_bound = int(stop * 0.5 * (1 - self.searching_ratio))
                right_bound = int(stop * 0.5 * (1 + self.searching_ratio))
                for min_idx, max_idx in zip(
                    range(0, left_bound, self.stride), range(stop - 1, right_bound, -self.stride)
                ):
                    if min_idx >= max_idx:
                        continue
                    max_value = max(distribution[max_idx], 0)
                    min_value = min(distribution[min_idx], 0)
                    mse = self._asymmetric_quantization_mse(valid_distribution, valid_bins, max_value, min_value)
                    if mse < min_mse:
                        min_mse = mse
                        min_values[chn] = float(min_value)
                        max_values[chn] = float(max_value)
        self.update_fake_quantizer_param_with_max_min(max_values, min_values)
        self.clear()

    def _symmetric_quantization_normalized_mse(self, valid_bins: np.ndarray, valid: np.ndarray, max_idx: int) -> float:
        """Compute the normalized MSE of the original histogram with symmetric quantization for a given max index.

        Find the normalized MSE of the original histogram when symmetric quantization is performed
        with a given max index. Compute the MSE when the interval of the histogram is 1.

        Args:
            valid_bins(np.ndarray) : histogram with a non-zero histogram value
            valid(np.ndarray) : boolean array storing whether the original histogram value was zero or not
            max_idx(int) :  maximum index that are not clipped

        Returns:
            Mean squared error of fakequantized histogram
        """
        nbins = self.quantizer.maxabs_bound + 1
        in_distribution = np.arange(0.5, max_idx)[valid[:max_idx]]
        fakequantized_in_distribution = (nbins * in_distribution / (max_idx - 0.5)).round() * (max_idx - 0.5) / nbins
        in_distribution_error = (in_distribution - fakequantized_in_distribution) ** 2
        out_distribution_error = (np.arange(1, len(valid) - max_idx + 1)[valid[max_idx:]]) ** 2
        mse = (np.append(in_distribution_error, out_distribution_error) * valid_bins).mean()
        return mse

    def _asymmetric_quantization_mse(
        self, distribution: np.ndarray, histogram: np.ndarray, max_value: float, min_value: float
    ) -> float:
        """Find the MSE of the original histogram when asymmetric quantization is performed with a given min and max.

        Args:
            distribution(np.ndarray): representative values in the histogram
            histogram(np.ndarray): the original histogram of the distribution want to quantize
            max_value(float): maximum values that are not clipped
            min_value(float): minimum values that are not clipped

        Returns:
            Mean squared error of fakequantized histogram
        """
        scale = np.float32(max_value - min_value) / (self.quantizer.quant_max - self.quantizer.quant_min)
        zero_point = -(min_value / scale).round() + self.quantizer.quant_min
        if zero_point < self.quantizer.quant_min or zero_point > self.quantizer.quant_max:
            raise ValueError("The quantization range of zero_point has been exceeded")
        fakequantized_distribution = (
            ((distribution / scale).round() + zero_point).clip(self.quantizer.quant_min, self.quantizer.quant_max)
            - zero_point
        ) * scale
        distribution_error = (distribution - fakequantized_distribution) ** 2
        return (distribution_error * histogram).mean()
