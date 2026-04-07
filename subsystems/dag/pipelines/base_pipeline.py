
from subsystems.dag.core.executor import PipelineExecutor


class BasePipeline:
    def __init__(self, steps):
        self.executor = PipelineExecutor(steps)

    def run(self, data):
        return self.executor.run(data)
