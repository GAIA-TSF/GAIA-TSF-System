
class PipelineExecutor:
    def __init__(self, steps):
        print('[PipelineExecutor] Initialized')
        self.steps = steps 

    def register(self, name, pipeline):
        print(f'[PipelineExecutor] Registering {name}')
        self.registry[name] = pipeline

    def run(self, data):
        for step in self.steps: 
            data = step.run(data) 
        return data 
    