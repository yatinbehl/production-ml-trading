def expanding_window_splits(n_samples, initial_train_size, test_size):
    splits = []

    train_end = initial_train_size

    while train_end + test_size <= n_samples:
        train_indices = list(range(0, train_end))
        test_indices = list(
            range(train_end, train_end + test_size)
        )

        splits.append((train_indices, test_indices))

        train_end += test_size

    return splits
