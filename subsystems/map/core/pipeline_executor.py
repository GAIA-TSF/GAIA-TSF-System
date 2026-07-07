from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger("map.pipeline_executor")

Operation = Callable[[Any, list[Any], dict[str, Any], Any], Any]


class PipelineExecutor:
    """Execute configured MAP pipeline DAGs.

    The executor reads a pipeline definition from ``config.pipelines`` and runs
    named operation nodes in dependency order. Operation implementations are
    registered separately, keeping the pipeline topology configurable without
    embedding domain logic in the executor.
    """

    def __init__(self, config: Any, operations: Mapping[str, Operation]) -> None:
        self.config = config
        self.operations = operations

    def run(self, pipeline_name: str, initial_context: dict[str, Any] | None = None) -> Any:
        """Run a configured pipeline and return its configured output node."""
        pipeline_config = getattr(getattr(self.config, "pipelines", None), pipeline_name, None)
        if pipeline_config is None:
            raise ValueError(f"Missing pipeline configuration for '{pipeline_name}'.")

        dag_config = getattr(pipeline_config, "dag", None)
        nodes = getattr(dag_config, "nodes", None)
        output_node = getattr(dag_config, "output", None)
        if nodes is None or output_node is None:
            raise ValueError(f"Pipeline '{pipeline_name}' must define dag.nodes and dag.output.")

        context: dict[str, Any] = {"config": self.config, **(initial_context or {})}
        visiting: set[str] = set()
        visited: set[str] = set()

        def execute_node(node_name: str) -> Any:
            if node_name in context:
                return context[node_name]
            if node_name in visiting:
                raise ValueError(f"Pipeline '{pipeline_name}' contains a cycle at node '{node_name}'.")

            node_config = getattr(nodes, node_name, None)
            if node_config is None:
                raise KeyError(f"Pipeline '{pipeline_name}' references unknown node '{node_name}'.")

            op_name = getattr(node_config, "op", None)
            if not op_name:
                raise ValueError(f"Pipeline node '{node_name}' must define an operation name.")
            operation = self.operations.get(op_name)
            if operation is None:
                raise KeyError(f"Pipeline operation '{op_name}' is not registered.")

            visiting.add(node_name)
            input_names = list(getattr(node_config, "inputs", []) or [])
            resolved_inputs = [execute_node(input_name) for input_name in input_names]
            visiting.remove(node_name)

            logger.info("Executing MAP pipeline node '%s' with op '%s'", node_name, op_name)
            context[node_name] = operation(self.config, resolved_inputs, context, node_config)
            visited.add(node_name)
            return context[node_name]

        result = execute_node(str(output_node))
        logger.info("Completed MAP pipeline '%s' with %s executed nodes", pipeline_name, len(visited))
        return result
