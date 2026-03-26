def expanding_window_splits(n, look_back, horizon, folds=3):
    fold_size = n // (folds + 1)
    splits = []

    for k in range(1, folds + 1):
        train_end = fold_size * k
        test_end = train_end + fold_size

        train_idx = range(0, train_end - look_back - horizon)
        test_idx = range(train_end - look_back, test_end - look_back - horizon)

        splits.append((train_idx, test_idx))

    return splits
