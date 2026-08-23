import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import evaluate_classifier
from src.prepare_data import prepare_ml_data
from src.walk_forward import expanding_window_splits


DATA_FILE = "data/AAPL_ml.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    dates = pd.to_datetime(df["Date"])

    X, y = prepare_ml_data(df)

    valid_indices = X.index
    dates = dates.loc[valid_indices]

    splits = expanding_window_splits(
        n_samples=len(X),
        initial_train_size=1500,
        test_size=250,
    )

    results = []

    for split_number, (train_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        test_dates = dates.iloc[test_idx]

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression()),
            ]
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_classifier(
            y_test,
            predictions,
            probabilities,
        )

        majority_class = y_train.mode()[0]
        baseline_predictions = [majority_class] * len(y_test)

        baseline_accuracy = accuracy_score(
            y_test,
            baseline_predictions,
        )

        results.append(
            {
                "split": split_number,
                "test_start": test_dates.iloc[0].date(),
                "test_end": test_dates.iloc[-1].date(),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "baseline_accuracy": baseline_accuracy,
                "model_accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "true_negative": metrics["true_negative"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "true_positive": metrics["true_positive"],



                "roc_auc": metrics["roc_auc"],
            }
        )

    results_df = pd.DataFrame(results)

    print(results_df)
   
    print("\nConfusion matrix by split:")

    print(
        results_df[
            [
                "split",
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
            ]
        ]
    )

    print(
        "\nAverage baseline accuracy:",
        round(results_df["baseline_accuracy"].mean(), 3),
    )
    print(
        "Average model accuracy:",
        round(results_df["model_accuracy"].mean(), 3),
    )
    print(
        "Average precision:",
        round(results_df["precision"].mean(), 3),
    )
    print(
        "Average recall:",
        round(results_df["recall"].mean(), 3),
    )
    print(
        "Average ROC-AUC:",
        round(results_df["roc_auc"].mean(), 3),
    )


if __name__ == "__main__":
    main()
