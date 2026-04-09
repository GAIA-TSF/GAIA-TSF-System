

def train_model(model, X, y, config):
    print("[Trainer] Training model")
    model.fit(X, y)
    return model 
