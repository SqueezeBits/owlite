from types import TracebackType
from typing import Optional

from torch.nn.parallel import DataParallel, DistributedDataParallel

from owlite_core.logger import log

from .backend.fx.types import GraphModuleOrDataParallel
from .enums import OwLiteStatus
from .nn import FakeQuantizer


def prepare_for_calibration(model: GraphModuleOrDataParallel) -> None:
    """Create a calibrator and prepare calibration according to opt.

    Args:
        model(GraphModule): graph module to calibrate.
    """
    log.info("Preparing for calibration")  # UX
    for _, module in model.named_modules(remove_duplicate=True):
        if isinstance(module, (FakeQuantizer,)):
            module.disable()
            module.calibrator.prepare()
    log.info("All fake quantizers in the model are now ready for calibration")  # UX
    log.info("Calibrating the model")  # UX


def update_fake_quantizers(model: GraphModuleOrDataParallel) -> None:
    """Calculate step size and zero point using data of calibrator and enabling quantization

    Args:
        model(GraphModuleOrDataParallel): model to calibrate.
    """
    log.info("Updating fake quantizers based on collected data")
    for name, module in model.named_modules(remove_duplicate=True):
        if isinstance(module, (FakeQuantizer,)):
            module.calibrator.update()
            if module.step_size.abs().max() <= 0:
                log.error(
                    f"({name}) : The step sizes are all zero. Make sure the data is fed to the quantizer correctly"
                )
                continue
            if module.step_size.min() < 0:
                log.warning(
                    f"({name}) : The step size contains a negative number. Automatically changed to positive",
                    stacklevel=2,
                )
                module.step_size.data = module.step_size.data.abs()
            module.enable()
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model.module.meta["owlite_status"] = OwLiteStatus.CALIBRATED
    else:
        model.meta["owlite_status"] = OwLiteStatus.CALIBRATED
    log.info("Updated fake quantizers. Calibration finished")  # UX


class CalibrationContext:
    """ContextManager for calibration"""

    def __init__(self, model: GraphModuleOrDataParallel):
        self.model = model

    def __enter__(self) -> GraphModuleOrDataParallel:
        prepare_for_calibration(self.model)
        return self.model

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        update_fake_quantizers(self.model)


def calibrate(model: GraphModuleOrDataParallel) -> CalibrationContext:
    """Calibration is performed using the supplied data within a 'with' statement.
    Set the step_size and zero_point of the fake quantizers using the calibrator that the fake quantizers.

        with calibrate(model):
            ... # feed data to model and store information from it.
        ... # calculate fake quantizers step_sizes and zero_points


    Args:
        model: GraphModule or DataParallel model to calibrate.

    Returns:
        CalibrationContext
    """
    return CalibrationContext(model)
