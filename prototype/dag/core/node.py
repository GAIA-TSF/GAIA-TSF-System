
"""
DAG Node definition.

A node represents a single processing step in the DAG.
Each node:
- has a name
- wraps a module (your processing logic)
- declares dependencies (inputs)
- produces an output stored in a shared context
"""


class DAGNode:
    def __init__(self, name: str, op: str, module, inputs: list):
        """
        Initialize a DAG node.

        Args:
            name (str): Unique node identifier in the DAG
            op (str): Operation type (e.g. 'feature_engineering')
            module: Python object implementing `.run()`
            inputs (list): Names of upstream nodes this node depends on
        """
        self.name = name
        self.op = op
        self.module = module
        self.inputs = inputs
        self.output = None  # store result after execution

    def run(self, context: dict):
        """
        Execute the node.

        Args:
            context (dict): Shared dictionary storing outputs of all nodes

        Returns:
            Any: Output produced by the module
        """
        print(f"[DAG] Running node: {self.name} ({self.op})")


        # --- Collect inputs from context ---
        # Each dependency must already exist in context

        # If single input → pass directly
        if len(self.inputs) == 1:
            input_data = context[self.inputs[0]]
        else:
            # multiple inputs → wrap into DataContainer
            from subsystems.dag.core.data_model import DataContainer

            input_data = DataContainer(
                data={k: context[k] for k in self.inputs},
                metadata={}
            )

        # --- Execute module logic ---
        # Each module must implement: module.run(inputs)
        result = self.module.run(input_data)

        # --- Store result in context under node name ---
        context[self.name] = result
        self.output = result

        return result
