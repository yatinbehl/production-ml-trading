FEATURE_COLUMNS = [
    "daily_return",
    "momentum_5d",
    "volatility_5d",
]

TARGET_COLUMN = "target"


def prepare_ml_data(df):
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    clean_df = df.dropna(subset=required_columns).copy()

    X = clean_df[FEATURE_COLUMNS]
    y = clean_df[TARGET_COLUMN].astype(int)
    return X, y


def time_series_split(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test
