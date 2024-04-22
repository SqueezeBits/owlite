from enum import IntEnum


class BenchmarkStatus(IntEnum):
    """Benchmark job status."""

    IDLE = 0
    PRE_FETCHING = 1
    BENCHMARKING = 2
    BENCHMARK_DONE = 3
    FETCHING_ERR = -1
    TIMEOUT_ERR = -2
    BENCHMARK_ERR = -3
    WEIGHT_GEN_ERR = -5
    STATUS_NOT_FOUND = -999

    @property
    def in_progress(self) -> bool:
        """Whether the status indicates if the benchmark is in progress."""
        return self in (
            BenchmarkStatus.PRE_FETCHING,
            BenchmarkStatus.BENCHMARKING,
        )

    @property
    def failed(self) -> bool:
        """Whether the status indicates if the benchmark has failed."""
        return self.value < 0
