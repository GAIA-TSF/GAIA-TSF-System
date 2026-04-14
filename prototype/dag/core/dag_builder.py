"""
DAG Builder.

Responsible for:
- parsing YAML configuration
- instantiating nodes
- wiring dependencies

This is the bridge between:
CONFIG (YAML) -> EXECUTABLE GRAPH
"""

# subsystems.
from dag.core.node import DAGNode
from dag.core.registry import registry
from dag.core.config_adapter import ConfigAdapter


class DAGBuilder:
    def __init__(self, config: dict, pipeline_name: str):
        """
        Initialize builder.

        Args:
            config (dict): Raw YAML config
            pipeline_name (str): e.g. 'amd'
        """
        self.config = ConfigAdapter(config, variable=pipeline_name)
        self.pipeline_name = pipeline_name

    def build(self):
        """
        Build DAG from config.

        Returns:
            nodes (dict): {node_name: DAGNode}
            output_node (str): name of final output node
        """

        # --- Extract pipeline-specific configuration ---
        pipeline_cfg = getattr(self.config.pipelines, self.pipeline_name)
        dag_cfg = pipeline_cfg.dag

        nodes = {}

        # --- Unified access to configuration sections --- 
        var_cfg = getattr(self.config.variables, self.pipeline_name)
        # feature_cfg = getattr(self.config.features, self.pipeline_name)
        try:
            feature_cfg = getattr(self.config.features, self.pipeline_name)
        except AttributeError:
            raise ValueError(
                f"[Config Error] Missing feature config for '{self.pipeline_name}'. "
                f"Check YAML structure under 'feature_engineering'."
            )
        output_cfg = self.config.output

        # --- Create nodes from YAML --- 
        print(type(dag_cfg.nodes))
        print(vars(dag_cfg.nodes))
        nodes_dict = vars(dag_cfg.nodes)

        for node_name, node_def in nodes_dict.items():

            op_name = node_def.op       # operation type
            inputs = node_def.inputs    # dependencies

            # --- Get module class from registry --- 
            module_cls = registry.get(op_name)

            # --- Instantiate module with correct configuration ---
            # This is where dependency injection happens 
            # Better solution than if else? 
            if op_name == "feature_engineering":
                module = module_cls(var_cfg, feature_cfg, output_cfg)

            elif op_name == "cloud_masking":
                module = module_cls(self.config) 

            elif op_name == "masking":
                module = module_cls(var_cfg)

            elif op_name == "tensorization":
                module = module_cls(self.config)

            else:
                # fallback for generic modules
                module = module_cls()

            # --- Create DAG node ---
            node = DAGNode(
                name=node_name,
                op=op_name,
                module=module,
                inputs=inputs,
            )

            print(f"[DAGBuilder] Node created: {node_name} <- {inputs}")

            nodes[node_name] = node

        # --- Define final output node ---
        output_node = dag_cfg.output

        return nodes, output_node