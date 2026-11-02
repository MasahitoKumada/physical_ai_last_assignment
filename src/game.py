"""Playable three-shot robot curling with pretrained, online-corrected forecasts."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from physics import CurlingWorld, DT, ShotCommand
from predictor import GlideObservation, IceRegressor, OnlineIceEstimator, predict_board

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "report/game_curved_model.json"
DEMO_COMMANDS = (ShotCommand(0., 79., 0), ShotCommand(2., 78., -1), ShotCommand(-2., 80., 1))


@dataclass(frozen=True)
class GameConfig:
    seed: int = 20261101
    initial_friction: float | None = None
    drift: float | None = None
    curl: float | None = None


class GameSession:
    """Own the true environment; only kinematics reach the separate estimator."""
    def __init__(self, config=None, model=None, adaptive=True):
        self.config = config or GameConfig()
        self.model = model or IceRegressor.load(MODEL)
        self.estimator = OnlineIceEstimator(self.model, enabled=adaptive)
        rng = np.random.default_rng(self.config.seed)
        initial = float(rng.uniform(.04, .06)) if self.config.initial_friction is None else self.config.initial_friction
        drift = float(rng.uniform(-.05, .05)) if self.config.drift is None else self.config.drift
        self._curl = float(rng.uniform(.02, .04)) if self.config.curl is None else self.config.curl
        self._ice = tuple(float(np.clip(initial * (1 + turn * drift), .02, .09)) for turn in range(3))
        self.world = CurlingWorld(self._ice[0], self._curl)
        self.reset()

    def reset(self):
        self.world.friction, self.world.curl = self._ice[0], self._curl
        self.world.reset()
        self.estimator.reset()
        self.history = []
        self.sweep_seconds = 0.
        self._contact_since_sample = False
        self._last_sample_tick = -1
        self._previous_sweep = False
        self.completed = 0

    def launch(self, command: ShotCommand):
        before = self.world.observe_board()
        initial_estimate = asdict(self.estimator.estimate)
        self.world.start_shot(command)
        self.estimator.start_shot()
        self.sweep_seconds, self._contact_since_sample = 0., False
        self._last_sample_tick, self._previous_sweep = -1, False
        self.history.append({"command": asdict(command), "start_estimate": initial_estimate,
                             "board_before": asdict(before), "observations": [], "sweep_events": []})

    def step(self, sweeping=False):
        if self.world.phase not in ("DELIVERY", "SLIDING", "SETTLING"):
            return
        self.world.step(sweeping)
        if self.world.phase == "ERROR":
            raise RuntimeError(self.world.error)
        current = self.world.current
        if self.world.release is not None:
            effective_sweep = bool(self.world.sweeping)
            self.sweep_seconds += DT * effective_sweep
            if effective_sweep != self._previous_sweep:
                self.history[-1]["sweep_events"].append({"time": float(self.world.data.time), "active": effective_sweep})
                self._previous_sweep = effective_sweep
            self._contact_since_sample |= any(
                self.world.disk_ids[current] in (int(c.geom1), int(c.geom2))
                and int(c.geom1) in self.world.disk_ids and int(c.geom2) in self.world.disk_ids
                for c in self.world.data.contact)
            tick = round((self.world.data.time - self.world.release) / DT)
            if tick >= 50 and (tick - 50) % 20 == 0 and tick != self._last_sample_tick:
                self._last_sample_tick = tick
                velocity = self.world.velocities()[current]
                observation = GlideObservation(
                    float(self.world.data.time), tuple(map(float, self.world.positions()[current, :2])),
                    tuple(map(float, velocity[3:5])), float(velocity[2]),
                    held=bool(self.world.data.eq_active[self.world.grasps[current]]) or not self.world.alive[current],
                    contact=self._contact_since_sample, sweeping=effective_sweep, sweep_seconds=self.sweep_seconds)
                self.history[-1]["observations"].append(asdict(observation))
                self.estimator.observe(observation)
                self._contact_since_sample = False
        if self.world.phase in ("READY", "RESULT"):
            self.completed += 1
            self.history[-1].update(result=self.world.result(), end_estimate=asdict(self.estimator.estimate),
                                    update_count=self.estimator.updates)
            if self.world.phase == "READY":
                self.world.friction = self._ice[self.completed]

    def run_until_rest(self, sweep_schedule=None, callback=None):
        while self.world.phase in ("DELIVERY", "SLIDING", "SETTLING"):
            sweep = bool(sweep_schedule(self)) if sweep_schedule is not None else False
            self.step(sweep)
            if callback is not None:
                callback(self)
        return self.world.result()

    def forecast(self, command=None, include_trace=False):
        return predict_board(self.world.observe_board(), self.estimator.estimate, command, include_trace)

    def save_record(self, path):
        payload = {"schema_version": 1, "config": asdict(self.config), "adaptive": self.estimator.enabled,
                   "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                   "sweep_event_time_convention": "timestamp at end of the integration step that applied the control",
                   "history": self.history, "estimate_updates": self.estimator.events,
                   "final_result": self.world.result()}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def verify_replay(path):
    record = json.loads(Path(path).read_text())
    if record.get("model_sha256", hashlib.sha256(MODEL.read_bytes()).hexdigest()) != hashlib.sha256(MODEL.read_bytes()).hexdigest():
        raise ValueError("Replay model differs from the saved record.")
    session = GameSession(GameConfig(**record["config"]), adaptive=record["adaptive"])
    errors = []
    for row in record["history"]:
        if "result" not in row:
            break  # A user may close the window during an unfinished throw.
        session.launch(ShotCommand(**row["command"]))
        event_index, sweeping = 0, False
        events = row["sweep_events"]
        while session.world.phase in ("DELIVERY", "SLIDING", "SETTLING"):
            while event_index < len(events) and events[event_index]["time"] - DT <= session.world.data.time + 1e-9:
                sweeping = events[event_index]["active"]
                event_index += 1
            session.step(sweeping)
        actual = session.world.result()
        for first, second in zip(actual["stones"], row["result"]["stones"]):
            if first["id"] != second["id"] or first["alive"] != second["alive"]:
                raise RuntimeError("Replay changed stone identity or in/out classification.")
            errors.append(float(np.linalg.norm(np.array(first["position"]) - second["position"])))
        if actual["score"] != row["result"]["score"]:
            raise RuntimeError("Replay changed a score.")
    maximum = max(errors, default=0.)
    if maximum > 1e-6:
        raise RuntimeError(f"Replay endpoint mismatch: {maximum} m")
    result = {"record": Path(path).name, "complete_shots": session.completed,
              "final_score": session.world.result()["score"], "max_position_error_m": maximum, "passed": True}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("play", "gui-smoke", "evaluate", "compare", "video", "report", "check-replay",
                                           "playtest", "playtest-summary", "playtest-smoke"))
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--account", default=None, help="OmniCampus account name or ID for submission filenames.")
    parser.add_argument("--demo", action="store_true", help="Use the fixed ice settings shown in the video.")
    parser.add_argument("--record", type=Path, default=ROOT / "report/game_demo_replay.json")
    parser.add_argument("--study", type=Path, help="Existing playtest directory to resume or summarize.")
    args = parser.parse_args()
    if args.command in ("playtest", "playtest-smoke", "playtest-summary"):
        from playtest import run_playtest, summarize_study
        if args.command == "playtest-summary":
            if args.study is None:
                parser.error("playtest-summary requires --study DIRECTORY")
            print(summarize_study(args.study)["text"])
        else:
            run_playtest(args.study, seed=args.seed, automated=args.command == "playtest-smoke")
    elif args.command == "compare":
        from comparison import build_evidence
        build_evidence()
    elif args.command in ("play", "gui-smoke"):
        from poc import PocRenderer
        from ui import run_game_ui
        config = GameConfig(20261101, .052, -.05, .03) if args.demo else GameConfig(seed=args.seed)
        run_game_ui(PocRenderer, GameSession(config), smoke=args.command == "gui-smoke")
    elif args.command == "check-replay":
        verify_replay(args.record)
    else:
        import analysis
        if args.command == "evaluate":
            analysis.evaluate_games()
        elif args.command == "video":
            analysis.make_video()
        else:
            analysis.make_report(args.account)


if __name__ == "__main__":
    main()
