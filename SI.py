"""
ocr_popup_handler.py
---------------------
Runs LOCALLY on your machine (needs a real screen - this will not run in a
sandbox/container with no display). Takes a screenshot, OCRs it, and only
acts when the recognized text matches something in YOUR predefined
instruction list below. Anything not in that list is logged and ignored -
this deliberately does NOT try to recognize or click "any popup" generically.

WHY IT WORKS THIS WAY:
A generic "find anything popup-shaped and click it" tool can't tell a
harmless toast notification apart from a permission prompt, a UAC/admin
dialog, a delete-confirmation, or a disguised malware/phishing popup -
clicking through those blindly is genuinely dangerous. Matching against
text YOU explicitly listed ahead of time avoids that: it only ever acts on
wording you've already reviewed and decided is safe to auto-dismiss.

OCR BACKEND: EasyOCR (no separate system binary to install, unlike
Tesseract - but it pulls in PyTorch, so the first install is a larger
download and the first run is slower while it loads the model).

Setup (run these on your own machine, not in a sandbox):
    pip install pyautogui easyocr pillow numpy
    # First run will download EasyOCR's English model (~100MB) automatically.
    # No separate OCR engine binary needed (that's the point vs. Tesseract).

Usage:
    python3 ocr_popup_handler.py                  # dry run (safe, default), single scan
    python3 ocr_popup_handler.py --watch           # dry run, scans every few seconds
    python3 ocr_popup_handler.py --live            # actually click matches
    python3 ocr_popup_handler.py --live --yes      # skip the confirmation prompt
    python3 ocr_popup_handler.py --live --watch    # actually click, continuously

HOW TO ADD YOUR OWN RULES:
Edit PREDEFINED_INSTRUCTIONS below. Each entry matches on exact substring
text (case-insensitive) found anywhere on screen, and defines exactly what
to do when it's found. Nothing outside this list is ever acted on.
"""

import argparse
import sys
import time
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# PREDEFINED INSTRUCTIONS - the only things this script will ever act on.
# match_text: substring to look for in OCR'd screen text (case-insensitive)
# sequence: list of steps, run relative to the matched text's location
# ---------------------------------------------------------------------------
PREDEFINED_INSTRUCTIONS = [
    {
        "id": "dismiss_update_later",
        "match_text": "update later",
        "description": "Click the 'Update Later' button on the app update nag",
        "sequence": [
            {"do": "click_match"},
        ],
    },
    {
        "id": "close_cookie_banner",
        "match_text": "accept all cookies",
        "description": "Dismiss a cookie consent banner by accepting it",
        "sequence": [
            {"do": "click_match"},
        ],
    },
    # Add more entries here. Example template:
    # {
    #     "id": "my_rule",
    #     "match_text": "exact text to look for",
    #     "description": "what this does and why it's safe to automate",
    #     "sequence": [{"do": "click_match"}],
    # },
]

# Anything that looks anywhere close to these categories should NEVER be
# auto-matched, even by accident via a careless entry above. This is a
# sanity check, not a substitute for reviewing your own entries.
DANGEROUS_KEYWORDS = [
    "delete", "remove", "uninstall", "format", "erase", "wipe",
    "administrator", "admin access", "sudo", "root access",
    "allow access", "grant permission", "authorize", "confirm password",
    "credit card", "social security", "bank account",
]


@dataclass
class Detection:
    rule_id: str
    match_text: str
    ocr_text: str
    x: int
    y: int
    width: int
    height: int


def _check_rules_are_safe():
    """Refuse to run if a predefined rule's match text overlaps a dangerous keyword."""
    problems = []
    for rule in PREDEFINED_INSTRUCTIONS:
        lowered = rule["match_text"].lower()
        for kw in DANGEROUS_KEYWORDS:
            if kw in lowered:
                problems.append((rule["id"], kw))
    if problems:
        print("Refusing to run: some predefined rules match dangerous keywords.")
        for rule_id, kw in problems:
            print(f"  - rule '{rule_id}' matches text containing '{kw}'")
        print("Remove or rewrite these rules before running.")
        sys.exit(1)


_easyocr_reader = None  # lazy singleton - loading the model takes a few seconds


def _get_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("Loading EasyOCR model (first run downloads it - this can take a moment)...")
        _easyocr_reader = easyocr.Reader(["en"], verbose=False)
    return _easyocr_reader


def scan_screen(reader):
    """
    Take a screenshot and OCR it, returning a list of {text, x, y, w, h}
    boxes in screen pixel coordinates.
    """
    import pyautogui
    import numpy as np

    screenshot = pyautogui.screenshot()
    results = reader.readtext(np.array(screenshot))  # [(bbox, text, confidence), ...]

    boxes = []
    for bbox, text, confidence in results:
        text = text.strip()
        if not text:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        boxes.append({"text": text, "x": x, "y": y, "w": w, "h": h})
    return boxes


def find_matches(boxes):
    """Group adjacent OCR word-boxes into lines, then check against predefined rules."""
    # merge word-level boxes into rough lines by proximity (same row band)
    lines = []
    boxes_sorted = sorted(boxes, key=lambda b: (b["y"], b["x"]))
    for b in boxes_sorted:
        placed = False
        for line in lines:
            if abs(line["y"] - b["y"]) < max(line["h"], b["h"]):
                line["text"] += " " + b["text"]
                line["x2"] = max(line["x2"], b["x"] + b["w"])
                line["y2"] = max(line["y2"], b["y"] + b["h"])
                placed = True
                break
        if not placed:
            lines.append({
                "text": b["text"], "x": b["x"], "y": b["y"],
                "x2": b["x"] + b["w"], "y2": b["y"] + b["h"], "h": b["h"],
            })

    detections = []
    for line in lines:
        line_text_lower = line["text"].lower()
        for rule in PREDEFINED_INSTRUCTIONS:
            if rule["match_text"].lower() in line_text_lower:
                cx = (line["x"] + line["x2"]) // 2
                cy = (line["y"] + line["y2"]) // 2
                detections.append(Detection(
                    rule_id=rule["id"],
                    match_text=rule["match_text"],
                    ocr_text=line["text"].strip(),
                    x=cx, y=cy,
                    width=line["x2"] - line["x"],
                    height=line["y2"] - line["y"],
                ))
    return detections


def run_sequence(pyautogui, detection, sequence, live):
    for step in sequence:
        action = step["do"]
        if action == "click_match":
            print(f"    -> move_to({detection.x}, {detection.y}) + click()  "
                  f"# matched text: \"{detection.ocr_text}\"")
            if live:
                pyautogui.moveTo(detection.x, detection.y, duration=0.25)
                pyautogui.click()
        elif action == "wait":
            secs = step.get("seconds", 0.3)
            print(f"    -> wait({secs}s)")
            if live and secs > 0:
                time.sleep(secs)
        elif action == "key":
            print(f"    -> key('{step['key']}')")
            if live:
                pyautogui.press(step["key"])
        else:
            print(f"    -> [unknown instruction '{action}', skipped]")


def do_one_scan(live: bool):
    try:
        import easyocr  # noqa: F401
    except ImportError:
        print("easyocr is required: pip install easyocr")
        sys.exit(1)

    try:
        import pyautogui  # noqa: F401
    except ImportError:
        print("pyautogui is required: pip install pyautogui")
        sys.exit(1)

    reader = _get_reader()
    boxes = scan_screen(reader)
    detections = find_matches(boxes)

    if not detections:
        print("No predefined-instruction matches found on screen this scan.")
        return

    print(f"{len(detections)} match(es) found:\n")
    for i, det in enumerate(detections, 1):
        rule = next(r for r in PREDEFINED_INSTRUCTIONS if r["id"] == det.rule_id)
        print(f"[{i}/{len(detections)}] rule='{det.rule_id}' "
              f"({rule['description']})")
        print(f"    matched on-screen text: \"{det.ocr_text}\" at ({det.x}, {det.y})")
        run_sequence(pyautogui, det, rule["sequence"], live)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="OCR-driven popup handler - only acts on predefined text matches."
    )
    parser.add_argument("--live", action="store_true",
                         help="Actually move the mouse and click (default: dry run only)")
    parser.add_argument("--yes", action="store_true",
                         help="Skip the confirmation prompt before --live actions")
    parser.add_argument("--watch", action="store_true",
                         help="Keep scanning every few seconds instead of a single scan")
    parser.add_argument("--interval", type=float, default=4.0,
                         help="Seconds between scans in --watch mode (default: 4.0)")
    args = parser.parse_args()

    _check_rules_are_safe()

    print(f"Loaded {len(PREDEFINED_INSTRUCTIONS)} predefined instruction(s): "
          + ", ".join(r["id"] for r in PREDEFINED_INSTRUCTIONS))
    print(f"Mode: {'LIVE (will actually click)' if args.live else 'DRY RUN (no real clicks)'}")

    if args.live and not args.yes:
        print("\nRules that will be LIVE-clicked if matched:")
        for r in PREDEFINED_INSTRUCTIONS:
            print(f"  - '{r['id']}': \"{r['match_text']}\" -> {r['description']}")
        resp = input("\nType 'yes' to continue: ")
        if resp.strip().lower() != "yes":
            print("Aborted.")
            return

    if args.watch:
        print(f"\nWatching screen every {args.interval}s. Ctrl+C to stop.\n")
        try:
            while True:
                do_one_scan(args.live)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print()
        do_one_scan(args.live)


if __name__ == "__main__":
    main()
