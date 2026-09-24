"""
main.py
-------
End-to-end pipeline that does actual work:

  image classifier  --\
                        --> priority score per event --> decentralized swarm
  text classifier   --/                                  allocation (emergent)
                                                                |
                                                                v
                                                    coverage / response-time report

The success criterion isn't "the swarm looks cool" - it's whether
high-priority events get responded to faster than low-priority ones,
using only local, decentralized decision-making (no central scheduler
ever sees the whole event list or assigns agents to events directly).

Run:
    python3 main.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from image_classifier import ImageClassifier
from text_classifier import TextClassifier
from events import generate_events
from swarm_allocation import ResponseSwarm
from metrics import coverage_report, print_report

N_EVENTS = 40
N_AGENTS = 6
BOUNDS = 20.0
MAX_STEPS = 400


def main():
    print("=" * 70)
    print("1) Training classifiers on real data")
    print("=" * 70)
    img_clf = ImageClassifier()
    img_acc = img_clf.train()
    print(f"Image classifier (digit/severity code) test accuracy: {img_acc:.3f}")

    txt_clf = TextClassifier()
    txt_acc = txt_clf.train()
    print(f"Text classifier (calm/neutral/urgent) train accuracy: {txt_acc:.3f}")

    print()
    print("=" * 70)
    print(f"2) Generating {N_EVENTS} events (real image + real text per event)")
    print("=" * 70)
    events = generate_events(N_EVENTS, img_clf, txt_clf, bounds=BOUNDS, seed=42)

    priorities = [ev.priority for ev in events]
    print(f"Priority range: {min(priorities):.3f} - {max(priorities):.3f} "
          f"(mean {sum(priorities)/len(priorities):.3f})")
    n_urgent_true = sum(1 for ev in events if ev.true_text_label == "urgent")
    n_urgent_correctly_classified = sum(
        1 for ev in events if ev.true_text_label == "urgent" and ev.text_pred_label == "urgent"
    )
    print(f"True urgent events: {n_urgent_true} "
          f"(classifier correctly flagged {n_urgent_correctly_classified})")

    print()
    print("=" * 70)
    print(f"3) Running decentralized allocation: {N_AGENTS} agents, "
          f"{N_EVENTS} events, {MAX_STEPS} max steps")
    print("=" * 70)
    print("Agents only see events within their sensor radius and never share")
    print("a global task list - coverage has to EMERGE from local decisions.")
    print()

    swarm = ResponseSwarm(
        n_agents=N_AGENTS,
        events=events,
        bounds=BOUNDS,
        sensor_radius=12.0,
        capture_radius=1.0,
        speed=1.2,
        seed=7,
    )
    result = swarm.run(max_steps=MAX_STEPS, dt=0.2)

    print()
    print("=" * 70)
    print("4) Results: did priority actually drive faster response?")
    print("=" * 70)
    report = coverage_report(events, result.steps_used)
    print_report(report)

    print()
    print("=" * 70)
    print("5) Saving visualization")
    print("=" * 70)
    _plot_results(events, result, report)
    print("Saved to allocation_results.png")


def _plot_results(events, result, report):
    traj = result.trajectory
    fig = plt.figure(figsize=(14, 5))

    # --- panel 1: final state, events colored by priority
    ax1 = fig.add_subplot(131, projection="3d")
    xs = [ev.position[0] for ev in events]
    ys = [ev.position[1] for ev in events]
    zs = [ev.position[2] for ev in events]
    colors = [ev.priority for ev in events]
    edge_colors = ["black" if ev.responded else "none" for ev in events]

    sc = ax1.scatter(xs, ys, zs, c=colors, cmap="YlOrRd", s=40, edgecolors=edge_colors)
    ax1.scatter(*traj[-1].T, c="blue", marker="^", s=80, label="agents (final)")
    ax1.set_title("Events (color=priority, black edge=responded)\n+ final agent positions")
    ax1.legend(loc="upper left", fontsize=7)
    fig.colorbar(sc, ax=ax1, shrink=0.6, label="priority")

    # --- panel 2: agent trajectories over time
    ax2 = fig.add_subplot(132, projection="3d")
    for a in range(traj.shape[1]):
        ax2.plot(*traj[:, a, :].T, linewidth=1)
    ax2.scatter(*traj[0].T, c="green", marker="o", s=40, label="start")
    ax2.scatter(*traj[-1].T, c="blue", marker="^", s=60, label="end")
    ax2.set_title(f"Agent trajectories over {result.steps_used} steps")
    ax2.legend(fontsize=7)

    # --- panel 3: response time by true urgency label (bar chart)
    ax3 = fig.add_subplot(133)
    by_label = report["response_time_by_true_urgency"]
    labels = ["urgent", "neutral", "calm"]
    avg_times = [by_label[l]["avg_response_time"] or 0 for l in labels]
    counts = [f"{by_label[l]['count_responded']}/{by_label[l]['count_total']}" for l in labels]
    bars = ax3.bar(labels, avg_times, color=["crimson", "goldenrod", "steelblue"])
    ax3.set_ylabel("avg response time (sim. time units)")
    ax3.set_title(f"Response time by true urgency\n"
                   f"coverage: {report['coverage_pct']:.0f}% "
                   f"(priority-weighted: {report['priority_weighted_coverage_pct']:.0f}%)")
    for bar, c in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  c, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("allocation_results.png", dpi=140)


if __name__ == "__main__":
    main()
