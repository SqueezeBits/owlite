# pylint: skip-file
# ruff: noqa: N806
# type: ignore
import numpy as np
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.tensor import Constant
from onnx_graphsurgeon.logger import G_LOGGER, LogMode
from onnx_graphsurgeon.util import misc


# Modified version of gs.Graph.fold_constants
# See the comments starting with "[SQZB]"
def fold_constants(
    graph: Graph,
    fold_shapes=True,
    recurse_subgraphs=True,
    partitioning=None,
    error_ok=True,
    flatten_subgraphs=True,
    size_threshold=None,
    should_exclude_node=None,
) -> Graph:
    """
    Folds constants in-place in the graph. The graph must be topologically sorted prior to
    calling this function (see `toposort()`).

    This function will not remove constants after folding them. In order to get rid of
    these hanging nodes, you can run the `cleanup()` function.

    *Note: Due to how this function is implemented, the graph must be exportable to ONNX,
    and evaluable in ONNX-Runtime. Additionally, ONNX-Runtime must be installed.*

    Args:
        fold_shapes (bool):
                Whether to fold `Shape` nodes in the graph.
                This requires shapes to be inferred in the graph, and can only fold
                static shapes.
                Defaults to True.
        recurse_subgraphs (bool):
                Whether to recursively fold constants in subgraphs.
                Defaults to True.
        partitioning (Union[str, None]):
                Whether/How to partition the graph so that errors in folding one
                part of a model do not affect other parts. Available modes are:

                - None: Do not partition the graph. If inference fails, no constants are folded.
                - "basic": Partition the graph. If inference fails in one partition, other partitions will
                        remain unaffected.
                - "recursive": Parition the graph recursively. If inference fails in a partition, the partition
                        will be further paritioned.

                Defaults to None.
        error_ok (bool):
                Whether inference errors should be suppressed.
                When this is False, any errors encountered during inference will be re-raised.
                Defaults to True.
        flatten_subgraphs (bool):
                Whether to flatten subgraphs where possible. For example, `If` nodes with a constant condition
                can be flattened into the parent graph.
        size_threshold (int):
                The maximum size threshold, in bytes, for which to fold constants.
                Any tensors larger than this value will not be folded.
                Set to ``None`` to disable the size threshold and always fold constants.
                For example, some models may apply ops like `Tile` or `Expand` to constants, which can
                result in very large tensors. Rather than pre-computing those constants and bloating
                the model size, it may be desirable to skip folding them and allow them to be computed
                at runtime.
                Defaults to None.
        should_exclude_node (Callable[[gs.Node], bool]):
                A callable that accepts an onnx-graphsurgeon node from the graph and reports whether it should
                be excluded from folding. This is only called for nodes which are otherwise foldable.
                Note that preventing a node from being folded also prevents its consumers from being folded.
                Defaults to a callable that always returns False.

    Returns:
        graph
    """
    from onnx_graphsurgeon.exporters.onnx_exporter import dtype_to_onnx, export_onnx

    should_exclude_node = misc.default_value(should_exclude_node, lambda node: False)

    PARTITIONING_MODES = [None, "basic", "recursive"]
    if partitioning not in PARTITIONING_MODES:
        G_LOGGER.critical(f"Argument for parameter 'partitioning' must be one of: {PARTITIONING_MODES}")
    ORT_PROVIDERS = ["CPUExecutionProvider"]

    G_LOGGER.debug(f"Folding constants in {graph.name}")

    # We apply constant folding in 5 passes:
    # Pass 1 lowers 'Constant' nodes into Constant tensors.
    # Pass 2 elides casts applied to shape tensors. This is done separately from other shape folding
    #   since it operates on the original graph rather than a clone.
    # Pass 3 finds all Constant tensors in the graph, then finds all descendants which are dependent
    #   only on constants.
    # Pass 4 searches for Shape nodes that have variable inputs (i.e. not marked const in pass 1)
    #    and turns them into Constants iff the input has a statically known shape.
    # Pass 5 computes the descendants determined in Pass 3 using ONNX-Runtime and replaces them in the graph.

    # Pass 1: Lower constant nodes
    for tensor in graph.tensors().values():
        if len(tensor.inputs) == 1:
            node = tensor.inputs[0]
            if node.op == "Constant":
                tensor.to_constant(node.attrs["value"]._values)  # Using ._values avoids copying
                tensor.inputs.clear()

    # Pass 2: Run shape-tensor cast elision
    def run_cast_elision(node):
        import onnx

        # Search for Casts (from int -> float) -> intermediate operator (with float constants) -> Casts (back to int)
        # This pattern is problematic for TensorRT since these operations may be performed on Shape Tensors, which
        # are not allowed to be floating point type. Attempt to fold the pattern here
        VALID_CAST_ELISION_OPS = [
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Max",
            "Min",
            "Equal",
            "Greater",
            "Less",
            "Concat",
        ]

        if node.op not in VALID_CAST_ELISION_OPS:
            return

        # If the uncasted outputs of this node have any consumers other than "Cast" nodes,
        # then we cannot elide the cast.
        for out_tensor in node.outputs:
            if out_tensor in graph.outputs:
                return

            if any(out_node.op != "Cast" for out_node in out_tensor.outputs):
                return

        # Get list of input nodes that cast to float32
        inp_casts = [
            inp_node
            for inp_tensor in node.inputs
            for inp_node in inp_tensor.inputs
            if inp_node.op == "Cast" and inp_node.attrs["to"] == onnx.TensorProto.DataType.FLOAT
        ]

        # [SQZB] Ensure that Cast nodes are attached to all of the input nodes.
        # Otherwise, onnx simplifier's shape inference could fail. (e.g. the test case "torch_f")
        if len(inp_casts) < len(node.inputs):
            return

        # No cast nodes found, return early
        if not inp_casts:
            return

        # Ensure that all input cast nodes are casting from the same type
        inp_dtypes = [dtype_to_onnx(inp_cast.inputs[0].dtype) for inp_cast in inp_casts]
        if len(set(inp_dtypes)) != 1:
            return

        final_type = inp_dtypes[0]

        # Get list of output nodes that cast to int32 or int64
        out_casts = [
            out_node
            for out_tensor in node.outputs
            for out_node in out_tensor.outputs
            if out_node.op == "Cast"
            and out_node.attrs["to"] in [onnx.TensorProto.DataType.INT32, onnx.TensorProto.DataType.INT64]
        ]

        # No cast node found on outputs, return early
        if not out_casts:
            return

        # Ensure that all output cast nodes are casting to the same type and that this
        # matches the original type before the inputs were casted.
        out_dtypes = [out_cast.attrs["to"] for out_cast in out_casts]
        if len(set(out_dtypes)) != 1 or out_dtypes[0] != final_type:
            return

        # If all checks passed, reconnect inputs/outputs to the consumers/producers
        # of the Cast nodes.
        # Note that we need to be careful in how we rebind tensors since they may
        # be used by multiple nodes. Thus, it is not necessarily safe to assume that
        # `cast_node.inputs[0].outputs[0] == cast_node`.
        for index, inp in enumerate(node.inputs):
            if isinstance(inp, Constant):
                inp.values = inp.values.astype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[final_type])

            for cast in inp_casts:
                if cast.outputs[0] == inp:
                    node.inputs[index] = cast.inputs[0]

        for index, out in enumerate(node.outputs):
            for cast in out_casts:
                if cast.inputs[0] == out:
                    out_tensor = cast.outputs[0]
                    out_tensor.inputs.clear()  # Disconnect from Cast
                    node.outputs[index] = out_tensor

    if fold_shapes:
        # Perform shape tensor cast elision prior to most other folding
        G_LOGGER.debug(f"Performing shape tensor cast elision in {graph.name}")
        try:
            with graph.node_ids():
                for node in graph.nodes:
                    run_cast_elision(node)
        except Exception as err:
            if not error_ok:
                raise err
            G_LOGGER.warning("'{:}' routine failed with: {:}".format("Shape tensor cast elision", err))

    # Note that most of the remaining passes operate on a clone of the original graph.
    # Pass 3: Find all descendants of constant tensors

    graph_clone = graph.copy()
    clone_tensors = graph_clone.tensors()

    def update_foldable_outputs(graph_constants):
        def is_foldable(node):
            NO_FOLD_OPS = [
                "QuantizeLinear",
                "DequantizeLinear",
                "DynamicQuantizeLinear",
            ]
            if node.op in NO_FOLD_OPS:
                return False

            def all_tensors_const(tensors):
                return all(t.name in graph_constants for t in tensors)

            if not all_tensors_const(node.inputs):
                return False

            all_subgraph_foreign_tensors_const = True
            for attr in node.attrs.values():
                if isinstance(attr, Graph):
                    foreign_tensors = attr._foreign_tensors().values()
                    all_subgraph_foreign_tensors_const &= all_tensors_const(foreign_tensors)

            return all_subgraph_foreign_tensors_const and not should_exclude_node(node)

        # Walks along the outputs of graph_constants to see if they can also be computed statically.
        # Since the graph is topologically sorted, this should find all constant nodes in the graph.
        for node in graph_clone.nodes:
            if is_foldable(node):
                graph_constants.update({out.name: out for out in node.outputs})
        return graph_constants

    graph_constants = {name: tensor for name, tensor in clone_tensors.items() if isinstance(tensor, Constant)}
    graph_constants = update_foldable_outputs(graph_constants)

    # Pass 4: Shape Folding

    def get_producer(tensor, op):
        """
        Get the producer of the specified tensor iff it matches op
        """
        if len(tensor.inputs) != 1:
            return None

        node = tensor.inputs[0]
        if node.op != op:
            return None
        return node

    def get_input(node, index=0):
        """
        Get the input tensor of a node iff the input tensor is not already marked a graph constant.
        """
        if node is None:
            return None

        inp = node.inputs[index]

        # If the input was already found to be a constant, it will be folded anyway.
        if inp.name in graph_constants:
            return None

        return inp

    def get_scalar_value(tensor):
        """
        Gets the scalar value of a constant tensor with a single item
        """
        if not tensor.shape:
            return tensor.values
        else:
            return list(tensor.values)[0]

    def fold_shape(tensor):
        inp = get_input(get_producer(tensor, "Shape"))
        if inp is None:
            return None

        if inp.shape is None or misc.is_dynamic_shape(inp.shape):
            return None
        return np.array(inp.shape, dtype=np.int64)

    def fold_shape_gather(tensor):
        gather = get_producer(tensor, "Gather")
        if gather is None:
            return None

        data = gather.inputs[0]
        indices_tensor = gather.inputs[1]

        inp = get_input(get_producer(data, "Shape"))
        if inp is None or inp.shape is None:
            return None

        if not isinstance(indices_tensor, Constant):
            return None

        indices = indices_tensor.values
        if not indices.shape:  # Scalar-case
            shape = inp.shape[int(indices)]
            if misc.is_dynamic_dimension(shape):
                return None
        else:
            shape = [inp.shape[index] for index in indices]
            if misc.is_dynamic_shape(shape):
                return None

        return np.array(shape, dtype=np.int64)

    def fold_shape_slice(tensor):
        slice = get_producer(tensor, "Slice")
        if slice is None:
            return None

        data = slice.inputs[0]

        if len(slice.inputs) >= 3:
            starts, ends = slice.inputs[1:3]
            if any(not isinstance(t, Constant) for t in [starts, ends]):
                return None
            starts, ends = get_scalar_value(starts), get_scalar_value(ends)
        elif "starts" in slice.attrs and "ends" in slice.attrs:
            starts, ends = slice.attrs["starts"][0], slice.attrs["ends"][0]
        else:
            return None

        inp = get_input(get_producer(data, "Shape"))
        if inp is None or inp.shape is None:
            return None

        # For shape tensors, we can only slice on the 0th dimension.
        if len(slice.inputs) > 3:
            axes = slice.inputs[3]
            if not isinstance(axes, Constant):
                return None

            if get_scalar_value(axes) != 0:
                return None
        elif "axes" in slice.attrs:
            if slice.attrs["axes"][0] != 0:
                return None

        steps = 1
        if len(slice.inputs) > 4:
            steps = slice.inputs[4]
            if not isinstance(steps, Constant):
                return None
            steps = get_scalar_value(steps)
        elif "steps" in slice.attrs:
            steps = slice.attrs["steps"][0]

        shape = inp.shape[starts:ends:steps]
        if misc.is_dynamic_shape(shape):
            return None

        return np.array(shape, dtype=np.int64)

    if fold_shapes:
        # NOTE: The order of shape folding passes is important to maximize how much we fold (phase-ordering problem).
        SHAPE_FOLD_FUNCS = [fold_shape_gather, fold_shape_slice, fold_shape]
        for shape_fold_func in SHAPE_FOLD_FUNCS:
            try:
                for tensor in clone_tensors.values():
                    shape_of = shape_fold_func(tensor)

                    if shape_of is not None:
                        G_LOGGER.ultra_verbose(f"Folding shape tensor: {tensor.name} to: {shape_of}")
                        graph_constants[tensor.name] = tensor.to_constant(shape_of)
                        graph_constants[tensor.name].inputs.clear()
            except Exception as err:
                if not error_ok:
                    raise err
                G_LOGGER.warning(f"'{shape_fold_func.__name__}' routine failed with:\n{err}")
            else:
                graph_constants = update_foldable_outputs(graph_constants)

    # Pass 5: Evaluate all tensors descended from constants with ONNX-Runtime and replace them with constant values.

    def partition_and_infer(subgraph):
        def get_out_node_ids():
            # Gets the final output nodes - producer nodes of graph output tensors without other outputs.
            with subgraph.node_ids():
                out_node_ids = set()
                for out in subgraph.outputs:
                    if not out.outputs and not isinstance(out, Constant):
                        for n_inp in out.inputs:
                            out_node_ids.add(subgraph._get_node_id(n_inp))
            return out_node_ids

        # Compute each output node in a separate subgraph.
        out_node_ids = get_out_node_ids()
        constant_values = {}

        for index in out_node_ids:  # Have to use index since 'node' is not in part
            part = subgraph.copy()
            out_node = part.nodes[index]
            part.outputs = out_node.outputs
            part.name = f"Folding: {[out.name for out in part.outputs]}"
            part.cleanup(remove_unused_graph_inputs=True)
            names = [out.name for out in part.outputs]

            try:
                # Determining types is not trivial, and ONNX-RT does its own type inference.
                import onnxruntime as onnxrt

                sess = onnxrt.InferenceSession(
                    export_onnx(part, do_type_check=False).SerializeToString(),
                    providers=ORT_PROVIDERS,
                )
                values = sess.run(names, {})
            except Exception as err:
                G_LOGGER.warning(f"Inference failed for subgraph: {part.name}. Note: Error was:\n{err}")
                if partitioning == "recursive":
                    G_LOGGER.verbose("Attempting to recursively partition subgraph")
                    # Partition failed, peel off last node.
                    # We only need to remove one node, so avoid doing an expensive call to cleanup()
                    part.outputs = out_node.inputs
                    del part.nodes[part.nodes.index(out_node)]
                    out_node.outputs.clear()
                    out_node.inputs.clear()
                else:
                    G_LOGGER.info("You may see better results if you set partitioning='recursive'")
                    if not error_ok:
                        raise err

                constant_values.update(partition_and_infer(part))
            else:
                constant_values.update(dict(zip(names, values)))

        return constant_values

    # Only evaluate foldable values that have non-foldable outputs or are graph outputs.
    # Otherwise, if all the outputs are foldable, then we can just evaluate the outputs directly.
    # Additionally, if we can determine tensor size, do not evaluate tensors whose sizes exceed the size threshold.
    def should_eval_foldable(tensor):
        from onnx_graphsurgeon.importers.onnx_importer import get_itemsize

        non_const = not isinstance(tensor, Constant)
        is_graph_output = not tensor.outputs
        has_non_foldable_outputs = any(out.name not in graph_constants for out in tensor.outputs)
        exceeds_size_threshold = (
            tensor.shape is not None
            and not misc.is_dynamic_shape(tensor.shape)
            and tensor.dtype is not None
            and size_threshold is not None
        ) and (misc.volume(tensor.shape) * get_itemsize(tensor.dtype) > size_threshold)

        return non_const and (is_graph_output or has_non_foldable_outputs) and not exceeds_size_threshold

    graph_clone.outputs = [t for t in graph_constants.values() if should_eval_foldable(t)]
    G_LOGGER.debug(f"Folding tensors: {graph_clone.outputs}")
    graph_clone.cleanup(remove_unused_graph_inputs=True)

    # Using ._values avoids a deep copy of the values.
    constant_values = {name: tensor._values for name, tensor in graph_constants.items() if isinstance(tensor, Constant)}
    if graph_clone.outputs:
        if partitioning:
            constant_values.update(partition_and_infer(graph_clone))
        else:
            names = [t.name for t in graph_clone.outputs]
            try:
                import onnxruntime as onnxrt

                sess = onnxrt.InferenceSession(
                    export_onnx(graph_clone, do_type_check=False).SerializeToString(),
                    providers=ORT_PROVIDERS,
                )
                values = sess.run(names, {})
                constant_values.update(dict(zip(names, values)))
            except Exception as err:
                G_LOGGER.warning(
                    "Inference failed. You may want to try enabling partitioning to see better results. "
                    "Note: Error was:\n{:}".format(err)
                )
                G_LOGGER.verbose(f"Note: Graph was:\n{graph_clone}")
                if not error_ok:
                    raise
    elif not constant_values:
        G_LOGGER.debug(
            "Could not find any nodes in this graph ({:}) that can be folded. "
            "This could mean that constant folding has already been run on this graph. "
            "Skipping.".format(graph.name)
        )

    # Finally, replace the Variables in the original graph with constants.
    large_tensors = {}
    if constant_values:
        graph_tensors = graph.tensors()
        for name, values in constant_values.items():
            tensor = graph_tensors[name]
            if isinstance(tensor, Constant):
                # No need to fold tensors that are already constant.
                continue

            if size_threshold is not None and values.nbytes > size_threshold:
                G_LOGGER.debug(
                    "Will not fold: '{:}' since its size in bytes ({:}) exceeds the size threshold ({:})".format(
                        name, values.nbytes, size_threshold
                    )
                )
                continue
            elif size_threshold is None and values.nbytes > (1 << 20):
                large_tensors[name] = values.nbytes

            tensor.to_constant(values)
            tensor.inputs.clear()  # Constants do not need inputs

        if large_tensors:
            large_tensors_mib = {
                tensor_name: f"{value // (1 << 20)} MiB" for tensor_name, value in large_tensors.items()
            }
            G_LOGGER.warning(
                "It looks like this model contains foldable nodes that produce large outputs.\n"
                "In order to avoid bloating the model, you may want to set a constant-folding size threshold.\n"
                "Note: Large tensors and their corresponding sizes were: {:}".format(large_tensors_mib),
                mode=LogMode.ONCE,
            )

    # Folding subgraphs after the outer graph can lead to better folding.
    def fold_subgraphs():
        for node in graph.nodes:
            for attr in node.attrs.values():
                if isinstance(attr, Graph):
                    attr.fold_constants(
                        fold_shapes=fold_shapes,
                        recurse_subgraphs=recurse_subgraphs,
                        partitioning=partitioning,
                        error_ok=error_ok,
                        flatten_subgraphs=flatten_subgraphs,
                        size_threshold=size_threshold,
                    )

    if recurse_subgraphs:
        fold_subgraphs()

    if flatten_subgraphs:
        # Flatten conditional subgraphs
        index = 0
        while index < len(graph.nodes):
            node = graph.nodes[index]
            if node.op == "If" and isinstance(node.inputs[0], Constant):
                G_LOGGER.debug(f"Flattening conditional: {node}")
                cond = get_scalar_value(node.inputs[0])
                subgraph = node.attrs["then_branch"] if cond else node.attrs["else_branch"]
                # Need to add a suffix to subgraph tensors so they don't collide with outer graph tensors
                for tensor in subgraph._local_tensors().values():
                    tensor.name += f"_subg_{index}_{subgraph.name}"

                # The subgraph outputs correspond to the If node outputs. Only the latter are visible
                # in the parent graph, so we rebind the producer nodes of the subgraph outputs to point
                # to the output tensors of the If instead.
                for node_out, subgraph_out in zip(node.outputs, subgraph.outputs):
                    node_out.inputs.clear()
                    for producer in subgraph_out.inputs:
                        for tensor_idx, out_tensor in enumerate(producer.outputs):
                            if out_tensor == subgraph_out:
                                producer.outputs[tensor_idx] = node_out

                # Copy subgraph nodes into parent graph at the index of the If.
                del graph.nodes[index]
                graph.nodes[index:index] = subgraph.nodes
                index += len(subgraph.nodes) - 1

            index += 1

    return graph
