""" Base Calibrator class"""


class _Calibrator:
    """Base Calibrator Class

    Uses the forward hook to collect the data needed for calibration and update the quantizer's
    step_size and zero_point.

    Attributes:
        hook_handler (Optional[torch.utils.hooks.RemovableHandle]): A hook handler.
        quantizer (FakeQuantizer): The `FakeQuantizer` to which the calibration will be applied.
    """

    def __init__(self, quantizer):
        self.hook_handler = None
        self.quantizer = quantizer

    def prepare(self):
        """Prepares calibration for the quantizer.

        Set temporal attributes on the quantizer and register a hook on the quantizer.

        Args:
            quantizer (FakeQuantizer): `FakeQuantizer` to calibrate.

        Raises:
            AttributeError: If the attribution to create in the quantizer already exists.

        Returns:
            torch.utils.hooks.RemovableHandle: A registered hook handler.
        """
        raise NotImplementedError

    def update(self):
        """Calculate step_size and zero_point of quantizer and update them. removes registered hook."""
        raise NotImplementedError
