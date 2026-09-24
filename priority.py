"""
priority.py
-----------
Fuses image + text classifier outputs into a single priority score in [0, 1]
for one "event" (e.g. a sensor reading + an incoming log message).

This replaces the earlier toy "control_signals" idea with something that
feeds directly into a real decision: which events get responded to first.
"""


def compute_priority(
    image_class: int,
    image_confidence: float,
    text_label: str,
    text_confidence: float,
) -> float:
    """
    image_class (0-9): treated as a severity code (higher = more severe).
    text_label: "calm" | "neutral" | "urgent".
    Confidences discount each signal toward a neutral 0.5 when the
    classifier itself is unsure - an uncertain classification shouldn't
    swing priority as hard as a confident one.
    """
    severity = image_class / 9.0
    severity_signal = image_confidence * severity + (1 - image_confidence) * 0.5

    urgency_map = {"calm": 0.1, "neutral": 0.5, "urgent": 1.0}
    urgency = urgency_map.get(text_label, 0.5)
    urgency_signal = text_confidence * urgency + (1 - text_confidence) * 0.5

    # weight text urgency slightly higher than image severity: an urgent
    # message about a low-severity reading still deserves fast response
    priority = 0.45 * severity_signal + 0.55 * urgency_signal
    return round(float(priority), 4)
