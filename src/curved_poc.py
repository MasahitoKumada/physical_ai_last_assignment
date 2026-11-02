"""Second learning gate: single-stone spin, curl, and executed sweeping.

Run: .venv/bin/python src/curved_poc.py run
The GUI and three-stone collision predictor are not connected by this gate.
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
from predictor import GlideObservation, IceEstimate, IceRegressor, ice_features, predict_free_glide
from learning_poc import metrics

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
MODEL = REPORT / "game_curved_model.json"
SEED = 20261101
ICE = {
    "train": [(mu, k) for mu in (.035, .04, .045, .05, .055, .06, .065) for k in (.015, .030, .045)],
    "validation": [(mu, k) for mu in (.037, .047, .057, .063) for k in (.020, .040)],
    "test": [(mu, k) for mu in (.0365, .0415, .0515, .0615) for k in (.022, .038)],
}
NOISE = (("clean", 0., 0., 0.), ("small_noise", .001, .005, .01), ("larger_noise", .005, .02, .05))


def sweep_active(mode, seconds):
    if not 0 <= seconds < .7 - 1e-9:
        return False  # Actual continuation matches the no-future-sweep forecast.
    if mode == "on":
        return True
    return mode == "pulse" and (.18 - 1e-9 <= seconds < .30 - 1e-9 or .46 - 1e-9 <= seconds < .58 - 1e-9)


def window_from(row):
    return tuple(GlideObservation(**o) for o in row["observations"])


def collect(split, regenerate=False):
    cache = REPORT / f"game_curved_{split}_shots.json"
    if cache.exists() and not regenerate:
        data = json.loads(cache.read_text())
        expected = [[mu, k] for mu, k in ICE[split]]
        if data.get("schema_version") != 1 or data["conditions"] != expected:
            raise ValueError(f"Incompatible data cache: {cache}. Choose a new artifact name.")
        print(f"{split}: using {len(data['shots'])} cached shots", flush=True)
        return data["shots"]
    rows = []
    for ice_index, (mu, kappa) in enumerate(ICE[split]):
        for spin_index, spin in enumerate((-1, 0, 1)):
            for sweep_index, mode in enumerate(("off", "on", "pulse")):
                power = (60., 74., 86.)[(ice_index + spin_index + sweep_index) % 3]
                angle = (-7., 0., 7.)[(ice_index + spin_index + 2 * sweep_index) % 3]
                command = ShotCommand(angle, power, spin)
                world = CurlingWorld(mu, kappa)
                world.start_shot(command)
                window, swept_seconds = [], 0.
                while world.phase not in ("READY", "RESULT", "ERROR"):
                    elapsed = -1. if world.release is None else float(world.data.time) - world.release
                    world.step(sweep_active(mode, elapsed))
                    if world.release is None:
                        continue
                    swept_seconds += DT * world.sweeping
                    ticks = round((float(world.data.time) - world.release) / DT)
                    if 50 <= ticks <= 350 and (ticks - 50) % 20 == 0:
                        velocity = world.velocities()[0]
                        contact = any(int(c.geom1) in world.disk_ids and int(c.geom2) in world.disk_ids for c in world.data.contact)
                        window.append(GlideObservation(
                            float(world.data.time), tuple(map(float, world.positions()[0, :2])),
                            tuple(map(float, velocity[3:5])), float(velocity[2]),
                            bool(world.data.eq_active[world.grasps[0]]), contact, world.sweeping, float(swept_seconds)))
                if world.phase == "ERROR" or len(window) != 16:
                    raise RuntimeError(f"{split}/{ice_index}: bad trajectory or observation window: {world.error}")
                feature = ice_features(tuple(window))
                rows.append({"id": f"{split}_{ice_index:02d}_{spin_index}_{sweep_index}", "true_friction": mu,
                             "true_curl": kappa, "command": asdict(command), "sweep_mode": mode,
                             "observations": [asdict(o) for o in window], "features": asdict(feature),
                             "release_time": world.release, "result": world.result()["stones"][0]})
        print(f"{split}: {ice_index + 1}/{len(ICE[split])} ice pairs", flush=True)
    cache.write_text(json.dumps({"schema_version": 1, "conditions": ICE[split], "mujoco_version": mujoco.__version__,
                                "shots": rows}, indent=2) + "\n")
    return rows


def train(regenerate=False):
    REPORT.mkdir(exist_ok=True)
    training, validation = collect("train", regenerate), collect("validation", regenerate)
    windows = [window_from(r) for r in training]
    mu, kappa = [r["true_friction"] for r in training], [r["true_curl"] for r in training]
    candidates = []
    for penalty in (1e-6, .01, 1., 10.):
        model = IceRegressor.fit(windows, mu, kappa, penalty)
        estimates = [model.estimate(window_from(r)) for r in validation]
        mu_mae = float(np.mean([abs(e.friction - r["true_friction"]) for e, r in zip(estimates, validation)]))
        curl_mae = float(np.mean([abs(e.curl - r["true_curl"]) for e, r in zip(estimates, validation) if e.curl_observable]))
        candidates.append({"penalty": penalty, "friction_mae": mu_mae, "observable_curl_mae": curl_mae,
                           "score": mu_mae / .03 + curl_mae / .03})
    selected = min(candidates, key=lambda row: row["score"])
    model = IceRegressor.fit(windows, mu, kappa, selected["penalty"])
    model.save(MODEL)
    payload = {"training_shots": len(training), "validation_shots": len(validation), "candidates": candidates,
               "selected": selected, "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest()}
    (REPORT / "game_curved_training.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def with_noise(window, position_std, velocity_std, spin_std, rng):
    return tuple(replace(o, position=tuple(np.array(o.position) + rng.normal(0., position_std, 2)),
                         velocity=tuple(np.array(o.velocity) + rng.normal(0., velocity_std, 2)),
                         angular_velocity=float(o.angular_velocity + rng.normal(0., spin_std))) for o in window)


def evaluate(regenerate=False):
    meta = json.loads((REPORT / "game_curved_training.json").read_text())
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != meta["model_sha256"]:
        raise ValueError("Model and training metadata differ; rerun train.")
    for axis in (0, 1):
        sets = [{pair[axis] for pair in ICE[split]} for split in ("train", "validation", "test")]
        assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i + 1, 3))
    test = collect("test", regenerate)
    model = IceRegressor.load(MODEL)
    prior = IceEstimate(model.prior_friction, model.prior_curl)
    rows, summaries, latencies, oracle_errors = [], {}, [], []
    for noise_index, (name, position_std, velocity_std, spin_std) in enumerate(NOISE):
        rng = np.random.default_rng(SEED + noise_index)
        group = []
        for index, shot in enumerate(test):
            clean_window = window_from(shot)
            window = with_noise(clean_window, position_std, velocity_std, spin_std, rng)
            started = time.perf_counter()
            estimate = model.estimate(window)
            learned = predict_free_glide(window[-1], estimate)
            latencies.append(time.perf_counter() - started)
            fixed = predict_free_glide(window[-1], prior)
            actual = shot["result"]
            row = {"id": shot["id"], "noise": name, "spin": shot["command"]["spin"], "sweep_mode": shot["sweep_mode"],
                   "true_friction": shot["true_friction"], "true_curl": shot["true_curl"],
                   "estimated_friction": estimate.friction, "estimated_curl": estimate.curl,
                   "curl_observable": estimate.curl_observable, "curl_weight": estimate.curl_weight,
                   "curl_standard_error": estimate.curl_standard_error, "actual_alive": actual["alive"],
                   "actual_position": actual["position"][:2], "fixed_alive": fixed["alive"], "learned_alive": learned["alive"],
                   "fixed_position": fixed["position"], "learned_position": learned["position"],
                   "fixed_endpoint_error_m": float(np.linalg.norm(np.array(fixed["position"]) - actual["position"][:2])),
                   "learned_endpoint_error_m": float(np.linalg.norm(np.array(learned["position"]) - actual["position"][:2]))}
            if name == "clean":
                friction_only = predict_free_glide(window[-1], IceEstimate(estimate.friction, prior.curl))
                row["friction_only_alive"] = friction_only["alive"]
                row["friction_only_endpoint_error_m"] = float(np.linalg.norm(np.array(friction_only["position"]) - actual["position"][:2]))
                # Oracle is strictly an evaluation diagnostic, not a model input.
                oracle = predict_free_glide(clean_window[-1], IceEstimate(shot["true_friction"], shot["true_curl"]))
                oracle_errors.append(float(np.linalg.norm(np.array(oracle["position"]) - actual["position"][:2])))
            group.append(row)
            if (index + 1) % 12 == 0:
                print(f"evaluate {name}: {index + 1}/{len(test)}", flush=True)
        observable = [r for r in group if r["curl_observable"]]
        summaries[name] = {"position_noise_std_m": position_std, "velocity_noise_std_m_s": velocity_std,
                           "angular_velocity_noise_std_rad_s": spin_std,
                           "friction_mae": float(np.mean([abs(r["estimated_friction"] - r["true_friction"]) for r in group])),
                           "observable_curl_mae": float(np.mean([abs(r["estimated_curl"] - r["true_curl"]) for r in observable])),
                           "observable_count": len(observable), "mean_curl_weight": float(np.mean([r["curl_weight"] for r in observable])),
                           "fixed": metrics(group, "fixed"), "learned": metrics(group, "learned"),
                           "by_sweep_mode": {mode: {method: metrics([r for r in group if r["sweep_mode"] == mode], method)
                                                       for method in ("fixed", "learned")} for mode in ("off", "on", "pulse")},
                           "by_spin": {str(spin): metrics([r for r in group if r["spin"] == spin], "learned") for spin in (-1, 0, 1)}}
        if name == "clean":
            summaries[name]["friction_only"] = metrics(group, "friction_only")
        rows.extend(group)
    clean = summaries["clean"]
    checks = {
        "friction_and_curl_values_disjoint_across_splits": True,
        "clean_oracle_restoration_error_under_1mm": max(oracle_errors) < .001,
        "clean_learned_mean_error_under_14cm": clean["learned"]["mean_stop_error_m"] < .14,
        "all_noise_groups_improve_over_fixed_prior": all(s["learned"]["mean_stop_error_m"] < s["fixed"]["mean_stop_error_m"] for s in summaries.values()),
        "clean_curl_learning_improves_over_friction_only": clean["learned"]["mean_stop_error_m"] < clean["friction_only"]["mean_stop_error_m"],
        "zero_spin_retains_curl_prior": all(not r["curl_observable"] and r["estimated_curl"] == prior.curl for r in rows if r["spin"] == 0),
        "nonzero_spin_observable": all(r["curl_observable"] for r in rows if r["spin"] != 0),
        "clean_all_sweep_modes_improve": all(g["learned"]["mean_stop_error_m"] < g["fixed"]["mean_stop_error_m"] for g in clean["by_sweep_mode"].values()),
        "clean_in_out_classification_perfect": clean["learned"]["in_out_accuracy"] == 1.,
        "inference_and_single_stone_rollout_under_1s": max(latencies) < 1.,
    }
    payload = {"status": "passed" if all(checks.values()) else "needs_review", "seed": SEED, "mujoco_version": mujoco.__version__,
               "scope": "single stone, spin/curl and executed sweeping; no future sweep",
               "prediction_time_after_release_seconds": .7, "observation_window_after_release_seconds": [.1, .7],
               "training_shots": meta["training_shots"], "validation_shots": meta["validation_shots"],
               "independent_test_shots": len(test), "noise_evaluations": len(rows), "conditions": ICE,
               "model_sha256": meta["model_sha256"], "checks": checks, "summaries": summaries,
               "max_oracle_error_m": max(oracle_errors), "max_prediction_seconds": max(latencies),
               "limitations": ["Same known drag, spin and sweep laws in source and forecast worlds; not real ice",
                               "Exact executed sweep history; synthetic independent Gaussian kinematic noise",
                               "Residual-based curl shrinkage is heuristic, not calibrated statistical confidence",
                               "Unseen friction and curl are inside training ranges; three noise groups reuse 72 test trajectories",
                               "Forecast assumes no sweep after 0.7 seconds; future user controls are not predicted",
                               "No collisions, three-shot persistence, pre-shot inference, or GUI integration"], "rows": rows}
    (REPORT / "game_curved_evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (REPORT / "game_curved_evaluation.csv").open("w", newline="") as file:
        fields = sorted(set().union(*(r.keys() for r in rows)))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot(payload)
    print(json.dumps({"status": payload["status"], "checks": checks,
                      "summary": {n: {"fixed_cm": s["fixed"]["mean_stop_error_m"] * 100,
                                      "learned_cm": s["learned"]["mean_stop_error_m"] * 100,
                                      "max_cm": s["learned"]["max_stop_error_m"] * 100,
                                      "curl_weight": s["mean_curl_weight"]} for n, s in summaries.items()},
                      "max_oracle_error_m": max(oracle_errors), "max_prediction_seconds": max(latencies)}, indent=2), flush=True)
    return payload


def plot(payload):
    cache = Path(tempfile.gettempdir()) / "curling-learning-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    groups = list(payload["summaries"].values())
    x = np.arange(len(groups))
    for offset, (method, label, color) in enumerate((("fixed", "Fixed ice prior", "#bc7452"), ("learned", "Learned ice", "#23899d"))):
        bars = axes[0].bar(x + (offset - .5) * .34, [s[method]["mean_stop_error_m"] * 100 for s in groups],
                           width=.34, color=color, label=label)
        axes[0].bar_label(bars, fmt=lambda value: "<0.01" if value < .01 else f"{value:.2f}", fontsize=9, padding=3)
    axes[0].set_xticks(x, ["Clean", "Small noise", "Larger noise"])
    axes[0].set_ylabel("Mean endpoint error for actual in-rink stops (cm)")
    axes[0].set_title("Single stone: left / no / right spin")
    axes[0].legend(fontsize=9)
    clean = payload["summaries"]["clean"]["by_sweep_mode"]
    for offset, (method, label, color) in enumerate((("fixed", "Fixed ice prior", "#bc7452"), ("learned", "Learned ice", "#23899d"))):
        bars = axes[1].bar(x + (offset - .5) * .34, [clean[mode][method]["mean_stop_error_m"] * 100 for mode in ("off", "on", "pulse")],
                          width=.34, color=color, label=label)
        axes[1].bar_label(bars, fmt=lambda value: "<0.01" if value < .01 else f"{value:.2f}", fontsize=9, padding=3)
    axes[1].set_xticks(x, ["No sweep", "Swept window", "Sweep toggles"])
    axes[1].set_ylabel("Mean endpoint error for actual in-rink stops (cm)")
    axes[1].set_title("Clean observations; no future sweeping")
    fig.savefig(REPORT / "game_curved_evaluation.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "run"))
    parser.add_argument("--regenerate", action="store_true", help="Recreate shot datasets instead of reusing cached observations.")
    args = parser.parse_args()
    if args.command in ("run", "train"):
        train(args.regenerate)
    if args.command in ("run", "evaluate"):
        if evaluate(args.regenerate)["status"] != "passed":
            raise SystemExit("Curved learning gate needs review; see saved evaluation.")


if __name__ == "__main__":
    main()
