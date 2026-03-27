
class PipelineExecutor:
    def __init__(self):
        print('[PipelineExecutor] Initialized')
        self.registry = {}

    def register(self, name, pipeline):
        print(f'[PipelineExecutor] Registering {name}')
        self.registry[name] = pipeline

    def run(self, name, inputs):
        print(f'[PipelineExecutor] Running {name}')
        return self.registry[name].run(inputs) 
    