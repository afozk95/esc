from __future__ import annotations
from typing import Dict, Generator, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class EnsemblesSpecializedClassifiers(BaseEstimator, ClassifierMixin):
    FINE_LABEL_VALUE_IN_DATASET_Y = 0
    OTHER_LABELS_VALUE_IN_DATASET_Y = 1

    def __init__(self, fine_to_coarse: Dict[str, str]) -> None:
        assert (
            isinstance(fine_to_coarse, dict) and
            all([isinstance(k, str) and isinstance(v, str) for k, v in fine_to_coarse.items()])
        ), "fine_to_coarse must be a Dict with str keys and str values (Dict[str, str])"
        self.coarse_labels: List[str] = list(set(fine_to_coarse.values()))
        self.fine_labels: List[str] = list(fine_to_coarse.keys())
        self.coarse_to_fine: Dict[str, List[str]] = self._make_coarse_to_fine(fine_to_coarse)
        self.fine_to_coarse: Dict[str, str] = fine_to_coarse
        self.coarse_labels_encoder: LabelEncoder = self._make_label_encoder(self.coarse_labels)
        self.fine_labels_encoder: LabelEncoder = self._make_label_encoder(self.fine_labels)
        self.fine_labels_models: Dict[int, Optional[ClassifierMixin]] = {fine_label_int: None for fine_label_int in self.fine_labels_encoder.transform(self.fine_labels)}

    def _make_coarse_to_fine(self, fine_to_coarse: Dict[str, str]) -> Dict[str, List[str]]:
        coarse_to_fine = defaultdict(list)
        {coarse_to_fine[v].append(k) for k, v in fine_to_coarse.items()}
        return coarse_to_fine

    def _make_label_encoder(self, labels: List[str]) -> LabelEncoder:
        le = LabelEncoder()
        le.fit(labels)
        return le

    def from_coarse_to_fine(self, coarse_to_fine: Dict[str, List[str]]) -> EnsemblesSpecializedClassifiers:
        assert (
            isinstance(coarse_to_fine, dict) and
            all([isinstance(k, str) for k in coarse_to_fine.keys()]) and
            all([isinstance(v, list) and all([isinstance(e, str) for e in v]) for v in coarse_to_fine.values()])
        ), "coarse_to_fine must be a Dict with str keys and List[str] values (Dict[str, List[str]])"
        fine_to_coarse = {}
        for k, v in coarse_to_fine.items():
            assert any(e in fine_to_coarse for e in v), f"duplicate fine label"
            fine_to_coarse.update({e: k for e in v})

        return EnsemblesSpecializedClassifiers(fine_to_coarse)

    def _make_datasets_for_fine_labels(self, X: np.ndarray, y: np.ndarray) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
        for fine_label in self.fine_labels:
            fine_label_int = self.fine_labels_encoder.transform([fine_label])[0]
            X_i = X
            y_i = np.where(y == fine_label_int, self.FINE_LABEL_VALUE_IN_DATASET_Y, self.OTHER_LABELS_VALUE_IN_DATASET_Y)
            yield fine_label_int, X_i, y_i

    def fit(self, X: np.ndarray, y: List[str], clf_default: Optional[ClassifierMixin] = None) -> EnsemblesSpecializedClassifiers:
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        y_fine = self.fine_labels_encoder.transform(y)
        datasets_fine = self._make_datasets_for_fine_labels(X, y_fine)
        for fine_label_int, X_i, y_i in datasets_fine:
            clf = RandomForestClassifier() if clf_default is None else clf_default
            clf.fit(X_i, y_i)
            self.fine_labels_models[fine_label_int] = clf
        return self

    def _make_pred_from_fine_labels_proba(self, fine_labels_pred_proba: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str], List[str]]:
        norm = "l1"  # "l1", "l2", "max"
        fine_labels_pred_proba = {fine_label_int: proba[:, self.FINE_LABEL_VALUE_IN_DATASET_Y] for fine_label_int, proba in fine_labels_pred_proba.items()}
        proba = np.stack(list(fine_labels_pred_proba.values()), axis=1)[:, list(fine_labels_pred_proba.keys())]
        pred_proba = normalize(proba, norm=norm, axis=1)
        pred_fine_labels = self.fine_labels_encoder.inverse_transform(np.argmax(pred_proba, axis=1))
        pred_coarse_labels = [self.fine_to_coarse[fine_label] for fine_label in pred_fine_labels]
        return pred_proba, pred_fine_labels, pred_coarse_labels

    def predict_proba(self, X) -> Tuple[np.ndarray, List[str], List[str]]:
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        fine_labels_pred_proba = {fine_label_int: model.predict_proba(X) for fine_label_int, model in self.fine_labels_models.items()}
        return self._make_pred_from_fine_labels_proba(fine_labels_pred_proba)
    
    def predict(self, X) -> Tuple[List[str], List[str]]:
        _, pred_y_fine, pred_y_coarse = self.predict_proba(X)
        return pred_y_fine, pred_y_coarse

    def score(self, X, y, sample_weight=None) -> Dict[str, float]:
        pred_y_fine, pred_y_coarse = self.predict(X)
        return {
            "accuracy": {
                "fine_labels": accuracy_score(y, pred_y_fine, sample_weight=sample_weight),
                "coarse_labels": accuracy_score([self.fine_to_coarse[e] for e in y], pred_y_coarse, sample_weight=sample_weight),
            }
        }
