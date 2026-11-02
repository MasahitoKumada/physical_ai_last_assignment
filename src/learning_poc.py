"""Train and evaluate the first, straight-shot friction-learning gate.

Run: .venv/bin/python src/learning_poc.py run
All artifacts use game_straight_*; the old model and GUI stay independently usable.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import mujoco
import numpy as np

from physics import CurlingWorld, DT, ShotCommand
from predictor import FrictionRegressor, GlideObservation, observed_deceleration, predict_straight_endpoint

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
MODEL_PATH = REPORT / "game_straight_model.json"
SEED = 20261031
TRAIN_ICE = tuple(round(.035 + .0025 * i, 8) for i in range(13))
VALIDATION_ICE = tuple(round(.036 + .005 * i, 8) for i in range(6))
TEST_ICE = tuple(round(.03675 + .0025 * i, 8) for i in range(12))
TRAIN_POWERS = (42, 50, 58, 66, 74, 79)
VALIDATION_POWERS = (46, 56, 68, 78)
TEST_POWERS = (45, 55, 65, 75, 82, 92)
SAMPLE_INTERVAL = .04
WINDOW_START = .10
WINDOW_END = .50
NOISE_CASES = (("clean", 0., 0.), ("small_noise", .001, .005), ("larger_noise", .005, .02))


def generate_shot(friction: float, power: float, identity: str) -> dict:
    """Truth is stored for supervised labels/evaluation, outside observations."""
    world = CurlingWorld(friction=friction)
    world.start_shot(ShotCommand(0., power, 0))
    observations = []
    every = round(SAMPLE_INTERVAL / DT)
    while world.phase not in ("READY", "RESULT", "ERROR"):
        world.step()
        if world.release is None:
            continue
        since_release = round((float(world.data.time) - world.release) / DT)
        if (round(WINDOW_START / DT) <= since_release <= round(WINDOW_END / DT)
                and (since_release - round(WINDOW_START / DT)) % every == 0):
            velocity = world.velocities()[0]
            contact = any(int(c.geom1) in world.disk_ids and int(c.geom2) in world.disk_ids for c in world.data.contact)
            observations.append(GlideObservation(
                time=float(world.data.time), position=tuple(map(float, world.positions()[0, :2])),
                velocity=tuple(map(float, velocity[3:5])), angular_velocity=float(velocity[2]),
                held=bool(world.data.eq_active[world.grasps[0]]), contact=contact, sweeping=world.sweeping))
    if world.phase == "ERROR":
        raise RuntimeError(f"{identity}: {world.error}")
    window = tuple(observations)
    if len(window) != 11:
        raise RuntimeError(f"{identity}: incomplete observation window ({len(window)} samples)")
    feature = observed_deceleration(window)
    return {"id": identity, "true_friction": float(friction), "power": float(power),
            "observations": [asdict(o) for o in window], "observed_deceleration": feature,
            "release_time": world.release, "result": world.result()["stones"][0]}


def window_from(shot: dict) -> tuple[GlideObservation, ...]:
    return tuple(GlideObservation(**row) for row in shot["observations"])


def collect(split: str, conditions, powers) -> list[dict]:
    rows = []
    for index, friction in enumerate(conditions):
        for number, power in enumerate(powers):
            rows.append(generate_shot(friction, power, f"{split}_{index:02d}_{number:02d}"))
        print(f"{split}: {index + 1}/{len(conditions)} ice conditions", flush=True)
    return rows


def train() -> dict:
    REPORT.mkdir(exist_ok=True)
    training = collect("train", TRAIN_ICE, TRAIN_POWERS)
    validation = collect("validation", VALIDATION_ICE, VALIDATION_POWERS)
    x = [r["observed_deceleration"] for r in training]
    y = [r["true_friction"] for r in training]
    candidates = []
    for penalty in (1e-6, .01, 1., 10.):
        candidate = FrictionRegressor.fit(x, y, penalty)
        mae = float(np.mean([abs(candidate.estimate(window_from(r)) - r["true_friction"]) for r in validation]))
        candidates.append({"regularization": penalty, "validation_friction_mae": mae})
    chosen = min(candidates, key=lambda c: c["validation_friction_mae"])
    model = FrictionRegressor.fit(x, y, chosen["regularization"])
    model.save(MODEL_PATH)
    payload = {"schema_version": 1, "seed": SEED, "mujoco_version": mujoco.__version__,
               "training": training, "validation": validation, "candidates": candidates,
               "chosen": chosen, "fixed_prior": float(np.mean(y)),
               "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()}
    (REPORT / "game_straight_training.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"training_shots": len(training), "validation_shots": len(validation),
                      "chosen": chosen}, indent=2), flush=True)
    return payload


def noisy_window(window, position_std, velocity_std, rng):
    return tuple(replace(o, position=tuple(np.array(o.position) + rng.normal(0., position_std, 2)),
                         velocity=tuple(np.array(o.velocity) + rng.normal(0., velocity_std, 2))) for o in window)


def metrics(rows: list[dict], name: str) -> dict:
    # Stops and rink exits are different outcomes: never call an exit a stop.
    stopped = [r for r in rows if r["actual_alive"]]
    errors = np.array([r[f"{name}_endpoint_error_m"] for r in stopped])
    return {"shots": len(rows), "actual_stops": len(stopped), "actual_exits": len(rows) - len(stopped),
            "mean_stop_error_m": float(errors.mean()), "median_stop_error_m": float(np.median(errors)),
            "p95_stop_error_m": float(np.quantile(errors, .95)), "max_stop_error_m": float(errors.max()),
            "stop_error_within_14cm_fraction": float(np.mean(errors <= .14)),
            "in_out_accuracy": float(np.mean([r[f"{name}_alive"] == r["actual_alive"] for r in rows])),
            "mean_endpoint_error_including_exits_m": float(np.mean([r[f"{name}_endpoint_error_m"] for r in rows]))}


def evaluate() -> dict:
    training = json.loads((REPORT / "game_straight_training.json").read_text())
    if training["model_sha256"] != hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest():
        raise ValueError("The model does not match the training record; rerun train.")
    model = FrictionRegressor.load(MODEL_PATH)
    train_conditions = {r["true_friction"] for r in training["training"]}
    val_conditions = {r["true_friction"] for r in training["validation"]}
    assert train_conditions.isdisjoint(val_conditions)
    assert set(TEST_ICE).isdisjoint(train_conditions | val_conditions)
    test = collect("test", TEST_ICE, TEST_POWERS)
    rows, summaries, latencies = [], {}, []
    for noise_index, (label, position_std, velocity_std) in enumerate(NOISE_CASES):
        rng = np.random.default_rng(SEED + noise_index)
        group = []
        for shot in test:
            window = noisy_window(window_from(shot), position_std, velocity_std, rng)
            before = time.perf_counter()
            estimate = model.estimate(window)
            learned = predict_straight_endpoint(window[-1], estimate)
            latencies.append(time.perf_counter() - before)
            fixed = predict_straight_endpoint(window[-1], training["fixed_prior"])
            actual = shot["result"]
            group.append({"id": shot["id"], "noise": label, "power": shot["power"],
                          "true_friction": shot["true_friction"], "estimated_friction": estimate,
                          "friction_error": abs(estimate - shot["true_friction"]),
                          "actual_alive": actual["alive"], "actual_x": actual["position"][0],
                          "observation_time_after_release": window[-1].time - shot["release_time"],
                          "fixed_alive": fixed["alive"], "learned_alive": learned["alive"],
                          "fixed_x": fixed["position"][0], "learned_x": learned["position"][0],
                          "fixed_endpoint_error_m": float(np.linalg.norm(np.array(fixed["position"]) - actual["position"][:2])),
                          "learned_endpoint_error_m": float(np.linalg.norm(np.array(learned["position"]) - actual["position"][:2]))})
        summaries[label] = {"position_noise_std_m": position_std, "velocity_noise_std_m_s": velocity_std,
                            "friction_mae": float(np.mean([r["friction_error"] for r in group])),
                            "fixed": metrics(group, "fixed"), "learned": metrics(group, "learned")}
        rows.extend(group)
    clean = summaries["clean"]
    checks = {
        "ice_conditions_disjoint_between_all_splits": True,
        "clean_stop_error_better_than_fixed": clean["learned"]["mean_stop_error_m"] < clean["fixed"]["mean_stop_error_m"],
        "clean_mean_stop_error_within_14cm": clean["learned"]["mean_stop_error_m"] < .14,
        "clean_in_out_classification_perfect": clean["learned"]["in_out_accuracy"] == 1.,
        "all_noise_groups_improve_mean_stop_error": all(s["learned"]["mean_stop_error_m"] < s["fixed"]["mean_stop_error_m"] for s in summaries.values()),
    }
    payload = {"status": "passed" if all(checks.values()) else "needs_review", "scope": "single straight shot, no sweep or collision",
               "prediction_time": "0.5 simulation seconds after physical robot release; not a pre-shot prediction",
               "forecast_method": "analytic constant Coulomb drag from observed position/velocity and estimated friction",
               "fixed_prior": training["fixed_prior"], "model_sha256": training["model_sha256"],
               "train_ice": sorted(train_conditions), "validation_ice": sorted(val_conditions), "test_ice": list(TEST_ICE),
               "training_shots": len(training["training"]), "validation_shots": len(training["validation"]),
               "independent_test_shots": len(test), "noise_evaluations": len(rows), "seed": SEED,
               "test_shots": test,
               "checks": checks, "summaries": summaries,
               "max_inference_seconds": max(latencies),
               "limitations": ["Same assumed friction law as the simulator; not real-ice validation",
                               "Unseen friction values are inside the training range",
                               "No rotation, curl, collisions, sweeping, or changing ice during a shot",
                               "No GUI integration or persistent game-time correction yet",
                               "Direct simulator velocity observations; no camera measurement",
                               "Noise is independent synthetic Gaussian noise, not calibrated sensor noise",
                               "Same test trajectories reused across noise levels; not independent extra shots",
                               "Analytic rollout differs slightly from the discrete simulator near rest and rink exit"],
               "rows": rows}
    (REPORT / "game_straight_evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (REPORT / "game_straight_evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(payload)
    print(json.dumps({k: payload[k] for k in ("status", "checks", "summaries", "max_inference_seconds")}, indent=2), flush=True)
    return payload


def plot(payload):
    cache = Path(tempfile.gettempdir()) / "curling-learning-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    names = list(payload["summaries"])
    x = np.arange(len(names))
    for offset, (method, label, color) in enumerate((("fixed", "Fixed friction prior", "#bc7452"),
                                                  ("learned", "Learned friction", "#23899d"))):
        means = [payload["summaries"][n][method]["mean_stop_error_m"] * 100 for n in names]
        bars = axes[0].bar(x + (offset - .5) * .34, means, width=.34, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    axes[0].set_xticks(x, ["No noise", "1 mm / 0.005 m/s", "5 mm / 0.02 m/s"])
    axes[0].set_ylabel("Mean stop-position error (cm)")
    axes[0].set_title("Held-out ice: actual in-rink stops only")
    axes[0].legend(fontsize=9)
    for noise, label, color in (("clean", "No noise", "#23899d"), ("larger_noise", "Larger observation noise", "#bc7452")):
        subset = [r for r in payload["rows"] if r["noise"] == noise]
        axes[1].scatter([r["true_friction"] for r in subset], [r["estimated_friction"] for r in subset],
                        s=18, alpha=.65, label=label, color=color)
    axes[1].plot([.035, .065], [.035, .065], "k--", lw=1, label="Perfect estimate")
    axes[1].set_xlabel("True friction (evaluation only)")
    axes[1].set_ylabel("Estimated friction")
    axes[1].set_title("Velocity window: 0.1-0.5 s after release")
    axes[1].legend(fontsize=9)
    figure.savefig(REPORT / "game_straight_evaluation.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "run"))
    args = parser.parse_args()
    if args.command in ("train", "run"):
        train()
    if args.command in ("evaluate", "run"):
        evaluation = evaluate()
        if evaluation["status"] != "passed":
            raise SystemExit("First learning gate needs review; see the saved evaluation.")


if __name__ == "__main__":
    main()
