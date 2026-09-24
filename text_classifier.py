"""
text_classifier.py
-------------------
A lightweight text classifier. Classifies short text into one of three
control-relevant categories: "calm", "neutral", "urgent". This mirrors how
a real system might turn free text (alerts, chat, commands) into a signal
that a control layer can act on.

Uses TF-IDF + Logistic Regression, trained on a small in-file labeled corpus
(no external downloads needed).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


TRAIN_TEXTS = [
    # calm
    "everything is fine, no action needed",
    "system running smoothly, all checks passed",
    "steady state maintained, nothing to report",
    "gentle breeze, quiet afternoon",
    "resting comfortably, low activity",
    "all systems nominal and stable",
    # neutral
    "processing request, please wait",
    "update available, review when convenient",
    "moderate load detected on the network",
    "new data received, awaiting classification",
    "task queued for standard processing",
    "status check scheduled for later today",
    # urgent
    "critical failure detected, immediate action required",
    "warning: temperature exceeding safe threshold",
    "emergency shutdown initiated, evacuate now",
    "alert! unauthorized access attempt detected",
    "system overload, respond immediately",
    "danger, collision imminent, brace for impact",
]

TRAIN_LABELS = (
    ["calm"] * 6 + ["neutral"] * 6 + ["urgent"] * 6
)


class TextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression(max_iter=1000)
        self._trained = False
        self.classes_ = None

    def train(self):
        X = self.vectorizer.fit_transform(TRAIN_TEXTS)
        self.model.fit(X, TRAIN_LABELS)
        self._trained = True
        self.classes_ = self.model.classes_
        # simple train accuracy as a sanity signal (tiny dataset, so expect ~1.0)
        return self.model.score(X, TRAIN_LABELS)

    def classify(self, text: str):
        """
        Returns: (predicted_label:str, confidence:float, full_prob_dict:dict)
        """
        if not self._trained:
            raise RuntimeError("Call .train() before .classify()")

        x = self.vectorizer.transform([text])
        probs = self.model.predict_proba(x)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.classes_[pred_idx]
        confidence = float(probs[pred_idx])
        prob_dict = dict(zip(self.classes_, probs.tolist()))
        return pred_label, confidence, prob_dict
