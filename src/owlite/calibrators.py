from types import TracebackType

import torch
from torch.fx.graph_module import GraphModule
from torch.nn.parallel import DataParallel, DistributedDataParallel
from tqdm import tqdm

from .backend.fx.types import GraphModuleOrDataParallel
from .core.constants import OWLITE_CALIBRATION_ENABLE_GRAD
from .core.logger import log
from .enums import ModelStatus
from .nn import FakeQuantizer


def _prepare_for_calibration(model: GraphModuleOrDataParallel) -> None:
    """Create a calibrator and prepare calibration according to opt.

    Args:
        model(`GraphModuleOrDataParallel`): graph module to calibrate.
    """
    log.info("Preparing for calibration")  # UX
    for _, module in model.named_modules(remove_duplicate=True):
        if isinstance(module, FakeQuantizer):
            module.disable()
            module.calibrator.prepare()
    log.info("All fake quantizers in the model are now ready for calibration")  # UX
    log.info("Calibrating the model")  # UX


def _update_fake_quantizers(model: GraphModuleOrDataParallel) -> None:
    """Calculate step size and zero point using data of calibrator and enabling quantization.

    Args:
        model(`GraphModuleOrDataParallel`): model to calibrate.
    """
    fake_quantizers = [m for m in model.modules() if isinstance(m, FakeQuantizer)]
    for module in tqdm(fake_quantizers, desc="Updating fake quantizers"):
        module.calibrator.update()
        if module.step_size.abs().max() <= 0:
            log.error(
                f"FakeQuantizer({module.id}) : The step sizes are all zero."
                "Make sure the data is fed to the quantizer correctly"
            )
            continue
        if module.step_size.min() < 0:
            log.warning(
                f"FakeQuantizer({module.id}) : The step size contains a negative number."
                "Automatically changed to positive",
                stacklevel=2,
            )
            module.step_size.data = module.step_size.data.abs()
        module.enable()
    if isinstance(model, DataParallel | DistributedDataParallel) and isinstance(model.module, GraphModule):
        model.module.meta["status"] = ModelStatus.CALIBRATED
    elif isinstance(model, GraphModule):
        model.meta["status"] = ModelStatus.CALIBRATED
    else:
        log.warning(
            "It looks like the model provided to `owlite.convert` is contaminated or have not created by the "
            "`OwLite.convert` method. The model might not be calibrated correctly."
        )  # UX
        return
    log.info("Updated fake quantizers. Calibration finished")  # UX


class CalibrationContext(torch.set_grad_enabled):
    """ContextManager for calibration.

    CalibrationContext disables gradient calculation.
    """

    def __init__(self, model: GraphModuleOrDataParallel):
        super().__init__(mode=OWLITE_CALIBRATION_ENABLE_GRAD)
        self.model = model

    def __enter__(self) -> GraphModuleOrDataParallel:  # type: ignore[override]
        super().__enter__()
        _prepare_for_calibration(self.model)
        return self.model

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            _update_fake_quantizers(self.model)
        super().__exit__(exc_type, exc_val, exc_tb)


def calibrate(model: GraphModuleOrDataParallel) -> CalibrationContext:
    """Calibration is performed using the supplied data within a 'with' statement.

    `owlite.calibrate` performs Post-Training Quantization (PTQ) calibration on a model converted with the
    `OwLite.convert`. It is required to preserve the model's accuracy by carefully selecting the quantization
    hyperparameters (the scale and zero-point). PTQ calibration typically requires only a subset of the training data.

    Please review the
    [Calibrator](https://squeezebits.gitbook.io/owlite/python-api/owlite.calibrators/owlite.calib.calibrator)
    for technical details.

    Args:
        model(`GraphModuleOrDataParallel`): GraphModule or DataParallel model to calibrate.

    Returns:
        CalibrationContext

    ### Usage

    `owlite.calibrate` returns an `owlite.CalibratorContext` object from the OwLite library can be used with a `with`
    statement to perform calibration. The `CalibratorContext` prepares the model for calibration and updates
    the model's fake quantizers after calibration is complete.

    **Example**

    ```python
    with owlite.calibrate(model):
        for i, data in enumerate(train_loader):
            model(*data) # feed data to model and store information from it.
        # calculate fake quantizers step_sizes and zero_points

    # You should use the `model` outside of the block after the calibration
    torch.save(model.state_dict())
    ```

    In this example, the `owlite.calibrate` creates an `owlite.CalibratorContext`,
    referenced by the variable `calibrator`. The training data fetched from `train_loader`
    are then passed to the `calibrator` to perform calibration.

    Note that you should continue writing your code outside of the `with` block since the fake quantizers
    in the model are updated as the `with` block exits.

    """
    return CalibrationContext(model)
