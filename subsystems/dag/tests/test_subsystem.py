
# -------------------------
# MAIN TEST RUN
# -------------------------

if __name__ == '__main__':
    executor = PipelineExecutor()

    executor.register('slope', SlopeStabilityPipeline())
    executor.register('amd', AMDPipeline())

    slope_output = executor.run('slope', {'s1': ['file1'], 'aoi': 'polygon'})
    print('Slope output:', slope_output)

    amd_output = executor.run('amd', {'s2': ['file2'], 'aoi': 'polygon', 'water_mask': 'mask'})
    print('AMD output:', amd_output)

