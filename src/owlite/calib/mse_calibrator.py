import numpy as np

from .histogram_calibrator import HistogramCalibrator


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

    def update(self) -> None:
        """Updates step_size using "`mse`"."""
        super().update()

        for chn, _ in enumerate(self.histc_bins):
            bins = self.histogram[chn].clone().detach().cpu().numpy().astype(np.int32)
            valid = bins != 0
            valid_bins = bins[valid]

            nbins = 1 << (int(self.quantizer.precision) - 1 + int(self.quantizer.unsigned))
            stop = len(bins)

            min_mse = np.inf
            last_argmin = stop

            for max_idx in range(512, stop + 1, 24):
                in_distribution = np.arange(0.5, max_idx)[valid[:max_idx]]
                in_distribution_error = (
                    in_distribution - (nbins * in_distribution / (max_idx - 0.5)).round() * (max_idx - 0.5) / nbins
                ) ** 2
                out_distribution_error = (np.arange(1, stop - max_idx + 1)[valid[max_idx:]]) ** 2

                mse = (np.append(in_distribution_error, out_distribution_error) * valid_bins).sum()

                if mse <= min_mse:
                    min_mse = mse
                    last_argmin = max_idx

            calib_amax = self.bin_edges[chn][last_argmin]
            self.quantizer.step_size.data[chn] = (calib_amax / self.quantizer.maxabs_bound).detach()

        self.clear()
