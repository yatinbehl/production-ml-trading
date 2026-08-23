import pytest

from src.evaluate import evaluate_classifier


def test_evaluate_classifier():
    y_true = [1, 1, 0, 0]
    predictions = [1, 0, 1, 0]
    probabilities = [0.8, 0.4, 0.7, 0.2]

    metrics = evaluate_classifier(
        y_true,
        predictions,
        probabilities,
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
