import numpy as np
import pytest

from src.bert.metrics import compute_accuracy, compute_confusion_matrix, compute_f1_score


@pytest.fixture
def labels_fixture():
    """Dummy labels for metric tests."""
    y_true = [0, 1, 2, 1, 0]
    y_pred = [0, 2, 2, 1, 0]
    return y_true, y_pred


class TestMetrics:
    """Test Metrics for bert"""

    def test_accuracy(self, labels_fixture):
        y_true, y_pred = labels_fixture
        actual = compute_accuracy(y_true, y_pred)

        assert actual == pytest.approx(0.8)

    def test_confusion_matrix(self, labels_fixture):
        """test compute confusion matrix"""
        y_true, y_pred = labels_fixture
        actual = compute_confusion_matrix(y_true, y_pred)

        expected = np.array([[2, 0, 0], [0, 1, 1], [0, 0, 1]])

        assert np.array_equal(actual, expected)

    def test_f1_score(self, labels_fixture):
        """Test F1 Score"""

        y_true, y_pred = labels_fixture

        actual = compute_f1_score(y_true, y_pred)

        assert actual == pytest.approx(0.8)
