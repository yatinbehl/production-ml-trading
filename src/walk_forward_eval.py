import pandas as pd

from sklearn.metrics import accuracy_score

from src.evaluate import evaluate_classifier
from src.models import (
    create_logistic_model,
    create_random_forest_model,
    get_random_forest_feature_importance,
)
from src.prepare_data import prepare_ml_data
from src.walk_forward import expanding_window_splits


pd.set_option("display.max_columns", None)

DATA_FILE = "data/AAPL_ml.csv"


def evaluate_model(
    model_name,
    model_factory,
    X,
    y,
    dates,
    splits,
):
    results = []
    feature_importance_results = []

    for split_number, (train_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        test_dates = dates.iloc[test_idx]

        model = model_factory()

        model.fit(X_train, y_train)

        if model_name == "Random Forest":
            importance = get_random_forest_feature_importance(
                model,
                X_train.columns,
            )

            print(
                f"\nRandom Forest feature importance - Split {split_number}:"
            )
            print(importance)

            for feature_name, importance_value in importance.items():
                feature_importance_results.append(
                    {
                        "split": split_number,
                        "feature": feature_name,
                        "importance": importance_value,
                    }
                )

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_classifier(
            y_test,
            predictions,
            probabilities,
        )

        majority_class = y_train.mode()[0]

        baseline_predictions = [
            majority_class
        ] * len(y_test)

        baseline_accuracy = accuracy_score(
            y_test,
            baseline_predictions,
        )

        results.append(
            {
                "model": model_name,
                "split": split_number,
                "test_start": test_dates.iloc[0].date(),
                "test_end": test_dates.iloc[-1].date(),
                "train_positive_rate": y_train.mean(),
                "test_positive_rate": y_test.mean(),
                "baseline_accuracy": baseline_accuracy,
                "model_accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "roc_auc": metrics["roc_auc"],
                "true_negative": metrics["true_negative"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "true_positive": metrics["true_positive"],
            }
        )

    return (
        pd.DataFrame(results),
        pd.DataFrame(feature_importance_results),
    )


def main():
    df = pd.read_csv(DATA_FILE)

    dates = pd.to_datetime(df["Date"])

    X, y = prepare_ml_data(df)

    dates = dates.loc[X.index]

    splits = expanding_window_splits(
        n_samples=len(X),
        initial_train_size=1500,
        test_size=250,
    )

    models = {
        "Logistic Regression": create_logistic_model,
        "Random Forest": create_random_forest_model,
    }

    all_results = []

    for model_name, model_factory in models.items():
        results, feature_importance = evaluate_model(
            model_name,
            model_factory,
            X,
            y,
            dates,
            splits,
        )

        all_results.append(results)

        if not feature_importance.empty:
            importance_summary = (
                feature_importance.groupby("feature")["importance"]
                .agg(["mean", "std"])
                .sort_values(
                    "mean",
                    ascending=False,
                )
            )

            print(
                "\nRandom Forest feature importance summary:"
            )
            print(importance_summary)

    results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    print("\nResults by model and split:")
    print(
        results_df[
            [
                "model",
                "split",
                "test_start",
                "test_end",
                "train_positive_rate",
                "test_positive_rate",
                "baseline_accuracy",
                "model_accuracy",
                "precision",
                "recall",
                "roc_auc",
            ]
        ]
    )

    print("\nAverage performance by model:")
    print(
        results_df.groupby("model")[
            [
                "baseline_accuracy",
                "model_accuracy",
                "precision",
                "recall",
                "roc_auc",
            ]
        ].mean()
    )


if __name__ == "__main__":
    main()
