"""
events.py
---------
Generates a batch of synthetic "events" - each one is a real digit image
(standing in for a sensor reading/severity code) paired with a real text
log message, placed at a random 3D location. Each event is run through
the trained classifiers to get a priority score.

This is the "meaningful data" side of the pipeline: every event has a
ground-truth position and a priority that downstream allocation must
respect.
"""

from dataclasses import dataclass, field
import numpy as np

from image_classifier import ImageClassifier
from text_classifier import TextClassifier, TRAIN_TEXTS, TRAIN_LABELS
from priority import compute_priority


@dataclass
class Event:
    id: int
    position: np.ndarray
    true_digit: int
    text: str
    true_text_label: str
    image_class: int = None
    image_confidence: float = None
    text_pred_label: str = None
    text_confidence: float = None
    priority: float = None
    responded: bool = False
    response_time: float = None
    responder_id: int = None


def generate_events(
    n_events: int,
    img_clf: ImageClassifier,
    txt_clf: TextClassifier,
    bounds: float = 20.0,
    seed: int = 0,
) -> list[Event]:
    rng = np.random.default_rng(seed)
    events = []

    for i in range(n_events):
        # pick a random severity code (digit) and a matching real image of it
        true_digit = int(rng.integers(0, 10))
        img_vec, _ = img_clf.sample_image(digit_class=true_digit, seed=seed * 1000 + i)

        # pick a real log message (with replacement) from the labeled corpus
        idx = rng.integers(0, len(TRAIN_TEXTS))
        text = TRAIN_TEXTS[idx]
        true_text_label = TRAIN_LABELS[idx]

        position = rng.uniform(-bounds / 2, bounds / 2, size=3)

        ev = Event(
            id=i,
            position=position,
            true_digit=true_digit,
            text=text,
            true_text_label=true_text_label,
        )

        # classify (the "meaningful work" - real inference, not ground truth)
        image_class, image_conf, _ = img_clf.classify(img_vec)
        text_label, text_conf, _ = txt_clf.classify(text)

        ev.image_class = image_class
        ev.image_confidence = image_conf
        ev.text_pred_label = text_label
        ev.text_confidence = text_conf
        ev.priority = compute_priority(image_class, image_conf, text_label, text_conf)

        events.append(ev)

    return events
