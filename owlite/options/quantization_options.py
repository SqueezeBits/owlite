from .fake_quantizer_options import FakeQuantizerOptions
from .options_dict import OptionsDict


class NodeQuantizationOptions(OptionsDict[str, FakeQuantizerOptions]):
    """
    * Key (str): the input node index or a predefined key. Numeric string keys are for inter-nodal modifications, while
        alphabetic string keys are for intra-nodal modifications.
    * Value (FakeQuantizerOptions): fake quantizer options
    """


class GraphQuantizationOptions(OptionsDict[str, NodeQuantizationOptions]):
    """
    * Key (str): the name of a FX node
    * Value (NodeQuantizationOptions): node quantization options
    """
