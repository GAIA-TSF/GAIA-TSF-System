"""
DAG Executor.

Responsible for:
- resolving dependencies
- executing nodes in correct order
- preventing deadlocks
"""


class DAGExecutor:
    def __init__(self, nodes: dict):
        """ 
        Args:
            nodes (dict): {node_name: DAGNode}
        """
        self.nodes = nodes

    def run(self, initial_context: dict):
        """
        Execute DAG.

        Args:
            initial_context (dict):
                Predefined inputs (e.g. {"input": data})

        Returns:
            dict: Full execution context with all node outputs
        """

        context = dict(initial_context)
        executed = set()

        print("[DAGExecutor] Starting execution")

        # --- Iterate until all nodes are executed --- 
        while len(executed) < len(self.nodes):

            progress = False

            for name, node in self.nodes.items():

                # skip already executed nodes
                if name in executed:
                    continue

                # --- Check if all dependencies are satisfied --- 
                if all(inp in context for inp in node.inputs):

                    node.run(context)

                    executed.add(name)
                    progress = True

            # --- Deadlock detection (cycle or missing input) --- 
            if not progress:
                missing = {
                    name: node.inputs
                    for name, node in self.nodes.items()
                    if name not in executed
                }
                raise RuntimeError(
                    f"[DAGExecutor] Execution stuck. Missing inputs: {missing}"
                )

        print("[DAGExecutor] Finished execution")

        return context
