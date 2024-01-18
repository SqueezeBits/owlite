from collections import defaultdict

from onnx import defs
from onnx.defs import ONNX_ML_DOMAIN, OpSchema


def get_full_operator_schemas() -> list[tuple[str, list[tuple[int, list[tuple[str, OpSchema, list[OpSchema]]]]]]]:
    """parse full operator schemas

    Returns:
        list[tuple[str, list[tuple[int, list[tuple[str, OpSchema, list[OpSchema]]]]]]]: nested structure containing all
            available op schemas
    """
    # domain -> support level -> name -> [schema]
    index: dict[str, dict[int, dict[str, list[OpSchema]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for schema in defs.get_all_schemas_with_history():
        index[schema.domain][int(schema.support_level)][schema.name].append(schema)

    # Preprocess the Operator Schemas
    # [(domain, [(support_level, [(schema name, current schema, all versions schemas)])])]
    operator_schemas: list[tuple[str, list[tuple[int, list[tuple[str, OpSchema, list[OpSchema]]]]]]] = []
    existing_ops: set[str] = set()
    for domain, _supportmap in sorted(index.items()):
        if domain == ONNX_ML_DOMAIN:
            continue

        processed_supportmap = []
        for _support, _namemap in sorted(_supportmap.items()):
            processed_namemap = []
            for n, unsorted_versions in sorted(_namemap.items()):
                versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                schema = versions[-1]
                if schema.name in existing_ops:
                    continue
                existing_ops.add(schema.name)
                processed_namemap.append((n, schema, versions))
            processed_supportmap.append((_support, processed_namemap))
        operator_schemas.append((domain, processed_supportmap))
    return operator_schemas


def get_core_operator_schemas() -> dict[str, OpSchema]:
    """restructured operator schemas for only core operators

    Returns:
        dict[str, list[tuple[int, list[tuple[str, OpSchema, list[OpSchema]]]]]]: the dictionary with key-value pairs
            where each op name is a key in string whose value is the nest structure containing various properties
            of the ONNX op.
    """
    triples = dict(get_full_operator_schemas())[""][0][1]
    return {x[0]: x[1] for x in triples}
