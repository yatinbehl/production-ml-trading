from src.walk_forward import expanding_window_splits


def test_expanding_window_splits():
    splits = expanding_window_splits(
        n_samples=100,
        initial_train_size=60,
        test_size=10,
    )

    assert len(splits) == 4

    first_train, first_test = splits[0]
    last_train, last_test = splits[-1]

    assert first_train[0] == 0
    assert first_train[-1] == 59
    assert first_test[0] == 60
    assert first_test[-1] == 69

    assert last_train[0] == 0
    assert last_train[-1] == 89
    assert last_test[0] == 90
    assert last_test[-1] == 99
