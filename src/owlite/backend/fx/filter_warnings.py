import linecache
import warnings
from typing import TextIO

from torch.jit._trace import TracerWarning


class FilterWarningsCausedByPasses(warnings.catch_warnings):
    """Context to ignore the warnings caused by FX optimization passes."""

    def __init__(self) -> None:
        super().__init__(record=False, module=warnings)

    def __enter__(self) -> None:
        warnings.showwarning = self.custom_showwarning
        return super().__enter__()

    def custom_showwarning(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        """Run `warnings.showwarning`, ignoring specific warnings caused by FX optimization passes.

        Args:
            message (Warning | str): the warning message to show
            category (type[Warning]): the warning category
            filename (str): the name of the file where this warning is caused
            lineno (int): the line number at which this warning is caused
            file (TextIO | None, optional): The file descriptor to write the warnings at. Defaults to None.
            line (str | None, optional): the line that caused this warning. Defaults to None.
        """
        if not hasattr(self, "_showwarning"):
            return
        if (
            category is TracerWarning
            and filename.startswith("<eval_with_key>")
            and "math_sqrt" in (line or linecache.getline(filename, lineno))
        ):
            # The FX optimization passes `DecomposeMultiHeadAttentionForward` and `DecomposeScaledDotProductAttention`
            # explicitly add nodes calling the function `math.sqrt`, generating a number of identical tracer warnings
            # such as: "<eval_with_key>.38:800: TracerWarning: Converting a tensor to a Python float might cause the
            # trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a
            # constant in the future. This means that the trace might not generalize to other inputs!
            #     sqrt_11 = math_sqrt(size_107);  size_107 = None"
            return
        msg = warnings.WarningMessage(message, category, filename, lineno, file, line)
        # pylint: disable-next=protected-access
        warnings._showwarnmsg_impl(msg)  # type: ignore[attr-defined]
