"""
metrics.py
----------
Turns raw simulation output into numbers that answer the actual question:
did the system do useful, priority-respecting work?
"""

import numpy as np


def coverage_report(events, steps_used: int):
    total = len(events)
    responded = [ev for ev in events if ev.responded]
    unresponded = [ev for ev in events if not ev.responded]

    total_priority = sum(ev.priority for ev in events)
    responded_priority = sum(ev.priority for ev in responded)

    report = {
        "total_events": total,
        "responded_events": len(responded),
        "coverage_pct": 100.0 * len(responded) / total if total else 0.0,
        "priority_weighted_coverage_pct": (
            100.0 * responded_priority / total_priority if total_priority > 0 else 0.0
        ),
        "steps_used": steps_used,
        "unresponded_ids": [ev.id for ev in unresponded],
    }

    # response time broken down by TRUE text urgency label -> the key
    # question: did urgent events genuinely get handled faster?
    by_label = {}
    for label in ("urgent", "neutral", "calm"):
        times = [ev.response_time for ev in responded if ev.true_text_label == label]
        by_label[label] = {
            "count_responded": len(times),
            "count_total": sum(1 for ev in events if ev.true_text_label == label),
            "avg_response_time": float(np.mean(times)) if times else None,
        }
    report["response_time_by_true_urgency"] = by_label

    # load balance across responder agents (std/mean of tasks per agent)
    responder_ids = [ev.responder_id for ev in responded if ev.responder_id is not None]
    if responder_ids:
        counts = np.bincount(responder_ids)
        counts = counts[counts >= 0]
        load_balance_cv = float(np.std(counts) / (np.mean(counts) + 1e-9))
    else:
        load_balance_cv = None
    report["load_balance_coefficient_of_variation"] = load_balance_cv

    return report


def print_report(report: dict):
    print(f"Total events:                {report['total_events']}")
    print(f"Responded:                   {report['responded_events']} "
          f"({report['coverage_pct']:.1f}% coverage)")
    print(f"Priority-weighted coverage:  {report['priority_weighted_coverage_pct']:.1f}%")
    print(f"Steps used:                  {report['steps_used']}")
    if report["unresponded_ids"]:
        print(f"Unresponded event ids:       {report['unresponded_ids']}")
    print()
    print("Response time by TRUE urgency label (lower = better; this is the")
    print("key check that priority actually drove behavior):")
    for label, stats in report["response_time_by_true_urgency"].items():
        avg = stats["avg_response_time"]
        avg_str = f"{avg:.2f}" if avg is not None else "n/a"
        print(f"  {label:8s}: {stats['count_responded']}/{stats['count_total']} "
              f"responded, avg time = {avg_str}")
    print()
    cv = report["load_balance_coefficient_of_variation"]
    if cv is not None:
        print(f"Load balance across agents (coeff. of variation): {cv:.3f} "
              f"(lower = more evenly shared)")
