"""
swarm_allocation.py
--------------------
The "emergence" layer, now doing actual work: a fleet of response agents
must cover as much priority-weighted event coverage as possible, as fast
as possible, using ONLY local rules:

  - each agent can only sense events within `sensor_radius` of itself
  - each agent picks its most attractive *visible* unclaimed event
    (attractiveness = priority / distance)
  - if two agents pick the same event in the same step, only the closer
    one keeps it; the other re-picks from what it can still see
  - an event is "responded to" once an agent gets within `capture_radius`

No agent has global knowledge of all events or all other agents' plans.
Good coverage of high-priority events emerges from these local rules,
not from central planning - that's the emergence principle doing real
work instead of just producing a visual pattern.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AllocationResult:
    trajectory: np.ndarray          # (steps, n_agents, 3)
    steps_used: int
    responded_count: int
    total_events: int


class ResponseSwarm:
    def __init__(
        self,
        n_agents: int,
        events,
        bounds: float = 20.0,
        sensor_radius: float = 12.0,
        capture_radius: float = 1.0,
        speed: float = 1.2,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.n_agents = n_agents
        self.events = events
        self.bounds = bounds
        self.sensor_radius = sensor_radius
        self.capture_radius = capture_radius
        self.speed = speed
        self.positions = rng.uniform(-bounds / 2, bounds / 2, size=(n_agents, 3))
        self.targets = [None] * n_agents  # event index each agent is currently pursuing

    def _visible_unclaimed(self, agent_idx: int):
        """Indices of unresponded events within this agent's sensor radius."""
        pos = self.positions[agent_idx]
        visible = []
        for j, ev in enumerate(self.events):
            if ev.responded:
                continue
            d = np.linalg.norm(ev.position - pos)
            if d <= self.sensor_radius:
                visible.append((j, d))
        return visible

    def _choose_targets(self):
        """Each agent locally picks its best visible event; resolve conflicts."""
        proposals = {}  # event_idx -> list of (agent_idx, distance)
        for i in range(self.n_agents):
            visible = self._visible_unclaimed(i)
            if not visible:
                self.targets[i] = None
                continue
            # attractiveness: priority matters more than small differences in
            # distance (priority is squared, distance is square-rooted) so
            # urgency genuinely wins races against merely-nearby events,
            # while extreme distance still eventually loses out
            best_j, best_score, best_d = None, -1.0, None
            for j, d in visible:
                score = (self.events[j].priority ** 2) / ((d + 1e-6) ** 0.5)
                if score > best_score:
                    best_j, best_score, best_d = j, score, d
            self.targets[i] = best_j
            proposals.setdefault(best_j, []).append((i, best_d))

        # conflict resolution: for each contested event, only the closest
        # agent keeps it; others drop their target this step (they'll
        # re-propose next step, possibly to a different event)
        for ev_idx, claimants in proposals.items():
            if len(claimants) > 1:
                claimants.sort(key=lambda x: x[1])
                for agent_idx, _ in claimants[1:]:
                    self.targets[agent_idx] = None

    def step(self, t: float, dt: float = 0.2):
        self._choose_targets()
        for i in range(self.n_agents):
            j = self.targets[i]
            if j is None:
                # no visible unclaimed event: gentle random walk to explore
                direction = np.random.default_rng().normal(size=3)
            else:
                direction = self.events[j].position - self.positions[i]

            norm = np.linalg.norm(direction) + 1e-6
            self.positions[i] += (direction / norm) * self.speed * dt
            self.positions[i] = np.clip(self.positions[i], -self.bounds, self.bounds)

            if j is not None:
                d = np.linalg.norm(self.events[j].position - self.positions[i])
                if d <= self.capture_radius and not self.events[j].responded:
                    self.events[j].responded = True
                    self.events[j].response_time = t
                    self.events[j].responder_id = i

    def run(self, max_steps: int = 400, dt: float = 0.2) -> AllocationResult:
        trajectory = np.zeros((max_steps, self.n_agents, 3))
        for step in range(max_steps):
            self.step(t=step * dt, dt=dt)
            trajectory[step] = self.positions
            if all(ev.responded for ev in self.events):
                trajectory = trajectory[: step + 1]
                return AllocationResult(
                    trajectory=trajectory,
                    steps_used=step + 1,
                    responded_count=len(self.events),
                    total_events=len(self.events),
                )

        responded = sum(1 for ev in self.events if ev.responded)
        return AllocationResult(
            trajectory=trajectory,
            steps_used=max_steps,
            responded_count=responded,
            total_events=len(self.events),
        )
