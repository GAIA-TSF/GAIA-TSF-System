class BasePipeline:
    def __init__(self):
        print('[BasePipeline] Initialized')

    def run(self, inputs):
        raise NotImplementedError
