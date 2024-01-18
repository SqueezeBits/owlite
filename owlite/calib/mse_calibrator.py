import torch

from .histogram_calibrator import HistogramCalibrator


class MSECalibrator(HistogramCalibrator):
    """MSE Calibrator Class"""

    def update(self) -> None:
        # update step_size using "MSE"
        super().update()

        # update step_size using "mse"
        for chn, _ in enumerate(self.histc_bins):
            centers = ((self.bin_edges[chn].data[1:] + self.bin_edges[chn].data[:-1]) / 2).to(
                self.quantizer.step_size.device
            )
            best_mse, best_arg = 1e10, -1

            for i in range(512, len(centers), 24):
                amax = centers[i]
                q_pos, q_neg = 2**self.quantizer.precision - 1, 0
                quant_centers = torch.fake_quantize_per_tensor_affine(centers, amax / q_pos, 0, q_neg, q_pos).to(
                    self.quantizer.step_size.device
                )
                mse = ((quant_centers - centers) ** 2 * self.histogram[chn].data).mean()
                if mse < best_mse:
                    best_mse, best_arg = mse, i
            mse_max = centers[best_arg : best_arg + 1]
            self.quantizer.step_size.data[chn] = (mse_max / self.quantizer.maxabs_bound).detach()

        self.clear_attribution()
