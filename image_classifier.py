"""
image_classifier.py
--------------------
A lightweight image "recogniser". Uses sklearn's built-in 8x8 digit dataset
(no external downloads needed) and a small MLP as a stand-in CNN.

Output is deliberately generic: a class label + a confidence vector, so it
can plug into any downstream control system.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ImageClassifier:
    def __init__(self, seed: int = 0):
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=400,
            random_state=seed,
        )
        self._trained = False

    def train(self):
        digits = load_digits()
        X, y = digits.data, digits.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model.fit(X_train, y_train)
        acc = self.model.score(X_test, y_test)
        self._trained = True
        return acc

    def classify(self, image_vector: np.ndarray):
        """
        image_vector: flat array matching digits.data shape (64,) for an 8x8 image.
        Returns: (predicted_class:int, confidence:float, full_prob_vector:np.ndarray)
        """
        if not self._trained:
            raise RuntimeError("Call .train() before .classify()")

        x = self.scaler.transform(image_vector.reshape(1, -1))
        probs = self.model.predict_proba(x)[0]
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])
        return pred, confidence, probs

    def sample_image(self, digit_class: int | None = None, seed: int = 0):
        """Grab a real sample image (optionally of a given class) for demo purposes."""
        digits = load_digits()
        rng = np.random.default_rng(seed)
        if digit_class is None:
            idx = rng.integers(0, len(digits.data))
        else:
            candidates = np.where(digits.target == digit_class)[0]
            idx = candidates[rng.integers(0, len(candidates))]
        return digits.data[idx], digits.target[idx]
