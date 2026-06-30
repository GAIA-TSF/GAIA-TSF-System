# from utils.io import save_model, save_config


"""
Learning pipeline

Unified training entry point
"""


def run_learning(config):
    """
    Unified training pipeline.

    Replaces:
        - lstm_learning
        - future RF/GBR scripts

    Steps:
        1. Select variable
        2. Load dataset
        3. Preprocess
        4. Feature engineering
        5. Train model
        6. Save artifacts
    """
    print('\n=== LEARNING PIPELINE ===')

    # 1. Select variable plugin
    # variable = VARIABLE_REGISTRY[config.variable]
    print(f'[Pipeline] Selected variable: {config.variable}')

    # 2. Load dataset (your existing dataset module)
    # data = load_dataset(config)
    print(f'[Pipeline] Dataset loaded: {config.dataset}')

    # 3. Variable-specific preprocessing
    # data = variable.preprocess(data, config)
    # Resolve active variable config
    var_cfg = getattr(config.variables, config.variable)

    # Access preprocessing
    preprocessing_cfg = var_cfg.preprocessing

    print(f'[Pipeline] Preprocessing config: {preprocessing_cfg}')

    # 4. Feature engineering
    # feature_fn = FEATURE_REGISTRY[variable.feature_pipeline()]
    # X, y = feature_fn(data, config)
    # how to define feature pipeline?
    print(f'[Pipeline] Feature engineering with pipeline: {config.features}')

    # 5. Validate model choice
    # assert config.model in variable.allowed_models(), \
    #     f"Model {config.model} not allowed for {config.variable}"
    print(f'[Pipeline] Selected model: {config.model}')

    # 6. Train model
    # model.fit(X, y)
    print(f'[Pipeline] Training model: {config.training}')

    # 7. Fine-tuning, validation, monitoring (if applicable)
    # if config.monitoring:
    #     monitor = Monitoring(config.monitoring)
    #     monitor.evaluate(model, X_val, y_val)
    print(f'[Pipeline] Monitoring with config: {config.monitoring}')

    # 8. Save model + config (reproducibility)
    # save_model(model, config)
    # save_config(config)
    print('[Pipeline] Saving model and config...')

    # return model
    return 0
