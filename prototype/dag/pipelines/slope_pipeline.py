
"""
Slope Pipeline using DAG execution.

This replaces the old linear pipeline with:
CONFIG → DAG → EXECUTION
"""

# subsystems. 
from dag.core.dag_builder import DAGBuilder
from dag.core.executor import DAGExecutor


class SlopePipeline:
    def __init__(self):
        self.config = None

    def run(self, input_data):
        """
        Run Slope pipeline.

        Args:
            input_data (dict): initial inputs from debug runner

        Returns:
            final output of DAG
        """
        # -------------------------------------------------------------
        # Build DAG from config
        # -------------------------------------------------------------
        builder = DAGBuilder(self.config, pipeline_name="slope")
        nodes, output_node = builder.build()

        # -------------------------------------------------------------
        # Create executor
        # -------------------------------------------------------------
        executor = DAGExecutor(nodes)

        # -------------------------------------------------------------
        # Initialize execution context
        # -------------------------------------------------------------
        context = {
            "input": input_data
        }

        # -------------------------------------------------------------
        # Run DAG
        # -------------------------------------------------------------
        result = executor.run(context)

        # -------------------------------------------------------------
        # Return final node output
        # -------------------------------------------------------------
        return result[output_node]
    
