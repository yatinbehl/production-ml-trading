import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score

from src.prepare_data import prepare_ml_data, time_series_split


DATA_FILE = "data/AAPL_ml.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    X, y = prepare_ml_data(df)

    X_train, X_test, y_train, y_test = time_series_split(
        X,
        y,
        train_ratio=0.8,
    )

    model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ]
)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nPrediction probabilities:")
    print(pd.Series(probabilities).describe())

    print("\nPrediction counts:")
    print(pd.Series(predictions).value_counts())

    print("\nActual test counts:")
    print(y_test.value_counts())

    model_accuracy = accuracy_score(y_test, predictions)

    model_auc = roc_auc_score(y_test, probabilities)

    majority_class = y_train.mode()[0]
    baseline_predictions = [majority_class] * len(y_test)

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    classifier = model.named_steps["classifier"]

    coefficients = pd.Series(
        classifier.coef_[0],
        index=X_train.columns,
    )

    print("\nModel coefficients:")
    print(coefficients.sort_values(ascending=False))

    print(f"Total usable rows: {len(X)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    print(f"\nMajority class: {majority_class}")
    print(f"Baseline accuracy: {baseline_accuracy:.3f}")
    print(f"Logistic regression accuracy: {model_accuracy:.3f}")

    print(f"Logistic regression ROC-AUC: {model_auc:.3f}")


if __name__ == "__main__":
    main()
