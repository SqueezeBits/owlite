from collections import Counter

import numpy as np

from .histogram_calibrator import HistogramCalibrator


# pylint: disable=too-many-locals
class EntropyCalibrator(HistogramCalibrator):
    r"""Entropy Calibrator Class
    The EntropyCalibrator compares the distribution of original data and
    the distribution of quantized data using KL divergence.
    When the original data $X$ and the quantized data $X_{quant}$ is given,
    the $step\_size$ is calculated as follow:
    $$
    step\_size = \underset {step\_size}{\operatorname{argmax}} \,
    KL \left( X || X_{quant} \right)
    $$
    This approach minimizes the divergence between two distributions.
    """

    def update(self) -> None:
        # update step_size using "entropy"
        super().update()

        for chn, _ in enumerate(self.histc_bins):
            bins = self.histogram[chn].clone().detach().cpu().numpy().astype(np.int32)
            bins[0] = bins[1]
            valid = bins != 0

            nbins = 1 << (self.quantizer.precision - 1 + int(self.quantizer.unsigned))
            stop = len(bins)

            min_divergence = np.inf
            last_argmin = stop

            for max_idx in range(512, stop + 1, 24):
                if not valid[max_idx - 1]:
                    continue
                valid_bins = bins[:max_idx][valid[:max_idx]]
                valid_digitized_space = (np.digitize(range(max_idx), np.linspace(0, max_idx, num=nbins + 1)) - 1)[
                    valid[:max_idx]
                ]

                new_density_counts = np.zeros(nbins, dtype=np.int32)
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

                if divergence <= min_divergence:
                    min_divergence = divergence
                    last_argmin = max_idx

            calib_amax = self.bin_edges[chn][last_argmin]
            self.quantizer.step_size.data[chn] = (calib_amax / self.quantizer.maxabs_bound).detach()

        self.clear()
