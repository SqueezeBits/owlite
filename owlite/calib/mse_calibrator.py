"""MSE(Mean Squared Error) calibrator"""
import torch

from ..logger import log
from ._histogram_calibrator import _HistogramCalibrator


class MSECalibrator(_HistogramCalibrator):
    """MSE Calibrator Class"""

    def update(self):
        # update step_size using "mse"
        if self.quantizer.histogram is None or self.quantizer.bin_edges is None:
            log.error(f"quantizer.histogram : {self.quantizer.histogram}")
            log.error(f"quantizer.bin_edges : {self.quantizer.bin_edges}")
            raise ValueError("quantizer.bin_edges or quantizer.histogram is None")

        for chn, _ in enumerate(self.quantizer.histc_bins):
            centers = (
                (
                    self.quantizer.bin_edges[chn].data[1:]
                    + self.quantizer.bin_edges[chn].data[:-1]
                )
                / 2
            ).to(self.quantizer.step_size.device)
            best_mse, best_arg = 1e10, -1

            for i in range(512, len(centers), 24):
                amax = centers[i]
                q_pos, q_neg = 2**self.quantizer.precision - 1, 0
                quant_centers = torch.fake_quantize_per_tensor_affine(
                    centers, amax / q_pos, 0, q_neg, q_pos
                ).to(self.quantizer.step_size.device)
                mse = (
                    (quant_centers - centers) ** 2 * self.quantizer.histogram[chn].data
                ).mean()
                if mse < best_mse:
                    best_mse, best_arg = mse, i
            mse_max = centers[best_arg : best_arg + 1]
            self.quantizer.step_size.data[chn] = (
                mse_max / self.quantizer.maxabs_bound
            ).detach()

        # delete "histogram" attritbution from quantizer
        for key in self.set_attr_list:
            delattr(self.quantizer, key)
        # remove registered forward_hook
        self.hook_handler.remove()
