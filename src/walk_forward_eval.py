import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.prepare_data import prepare_ml_data
from src.walk_forward import expanding_window_splits


DATA_FILE = "data/AAPL_ml.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    X, y = prepare_ml_data(df)

    splits = expanding_window_splits(
        n_samples=len(X),
        initial_train_size=1500,
        test_size=250,
    )

    results = []

    for split_number, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression()),
            ]
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

        results.append(
            {
                "split": split_number,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": accuracy,
                "roc_auc": auc,
            }
        )

    results_df = pd.DataFrame(results)

    print(results_df)

    print("\nAverage accuracy:", round(results_df["accuracy"].mean(), 3))
    print("Average ROC-AUC:", round(results_df["roc_auc"].mean(), 3))


if __name__ == "__main__":
    main()
