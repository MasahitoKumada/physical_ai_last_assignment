"""Evaluate observation-only, learned-ice forecasts of every stone after contact.

The previously trained curved model is frozen. All forecasts start 0.7 seconds
after release, before any contact, and assume no subsequent sweeping.
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

from physics import BoardObservation, CurlingWorld, DT, ShotCommand, StoneObservation
from predictor import GlideObservation, IceEstimate, IceRegressor, predict_board
from curved_poc import NOISE, sweep_active, with_noise, window_from

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
MODEL = REPORT / "game_curved_model.json"
SEED = 20261102
ICE = [(mu, kappa) for mu in (.038, .043, .053, .0635) for kappa in (.024, .036)]
CASES = ("free", "head_on", "grazing", "chain")


def fixture(case, mu=.05, kappa=.03):
    world = CurlingWorld(mu, kappa)
    if case == "head_on":
        world.place_fixture(1, (2., 0.))
    elif case == "grazing":
        world.place_fixture(1, (2., .115))
    elif case == "chain":
        world.place_fixture(1, (2., 0.))
        world.place_fixture(2, (2.22, 0.))
    elif case != "free":
        raise ValueError(case)
    return world


def board_from(payload):
    values = dict(payload)
    values["stones"] = tuple(StoneObservation(**{**s, "position": tuple(s["position"]), "velocity": tuple(s["velocity"])})
                              for s in values["stones"])
    for name in ("robot_position", "robot_velocity"):
        values[name] = tuple(values[name])
    if values["command"] is not None:
        values["command"] = ShotCommand(**values["command"])
    return BoardObservation(**values)


def source_hash():
    # Physics changes must never silently reuse old truth trajectories.
    return hashlib.sha256((ROOT / "src/physics.py").read_bytes() + (ROOT / "src/game_scene.xml").read_bytes()).hexdigest()


def collect(regenerate=False):
    cache = REPORT / "game_collision_shots.json"
    if cache.exists() and not regenerate:
        data = json.loads(cache.read_text())
        if data["physics_sha256"] != source_hash() or data["conditions"] != [list(pair) for pair in ICE]:
            raise ValueError("Physics/data changed; rerun with --regenerate.")
        print(f"using {len(data['shots'])} cached collision trials", flush=True)
        return data["shots"]
    rows = []
    for ice_index, (mu, kappa) in enumerate(ICE):
        for case_index, case in enumerate(CASES):
            for spin_index, spin in enumerate((-1, 0, 1)):
                power = (79., 86.)[(ice_index + spin_index) % 2]
                sweep_mode = ("off", "pulse")[(ice_index + spin_index + case_index) % 2]
                command = ShotCommand(0., power, spin)
                world = fixture(case, mu, kappa)
                initial = world.observe_board()
                world.start_shot(command)
                window, swept, board = [], 0., None
                while world.phase not in ("READY", "RESULT", "ERROR"):
                    elapsed = -1. if world.release is None else world.data.time - world.release
                    world.step(sweep_active(sweep_mode, elapsed))
                    if world.release is None:
                        continue
                    swept += DT * world.sweeping
                    tick = round((world.data.time - world.release) / DT)
                    if 50 <= tick <= 350 and (tick - 50) % 20 == 0:
                        velocity = world.velocities()[0]
                        window.append(GlideObservation(float(world.data.time), tuple(map(float, world.positions()[0, :2])),
                                                       tuple(map(float, velocity[3:5])), float(velocity[2]),
                                                       bool(world.data.eq_active[world.grasps[0]]), bool(world.collisions),
                                                       world.sweeping, float(swept)))
                    if tick == 350:
                        if world.collisions:
                            raise RuntimeError("Fixture collided before the prediction window ended.")
                        board = world.observe_board()
                if world.phase == "ERROR" or board is None or len(window) != 16:
                    raise RuntimeError(f"Invalid collision trial: {case}, {world.error}")
                rows.append({"id": f"{case}_{ice_index:02d}_{spin_index}", "case": case,
                             "true_friction": mu, "true_curl": kappa, "command": asdict(command), "sweep_mode": sweep_mode,
                             "initial_board": asdict(initial), "observations": [asdict(o) for o in window],
                             "board": asdict(board), "result": world.result()})
        print(f"collect: {ice_index + 1}/{len(ICE)} ice pairs", flush=True)
    cache.write_text(json.dumps({"schema_version": 1, "physics_sha256": source_hash(), "conditions": ICE,
                                "mujoco_version": mujoco.__version__, "shots": rows}, indent=2) + "\n")
    return rows


def perturb_board(board, window, position_std, velocity_std, spin_std, rng):
    stones = []
    for s in board.stones:
        if not s.alive:
            stones.append(s)
        elif s.id == board.current:
            # Same measurement as the estimator, not an extra noise draw.
            o = window[-1]
            stones.append(replace(s, position=o.position, velocity=o.velocity, angular_velocity=o.angular_velocity))
        else:
            stones.append(replace(s, position=tuple(np.asarray(s.position) + rng.normal(0., position_std, 2)),
                                  velocity=tuple(np.asarray(s.velocity) + rng.normal(0., velocity_std, 2)),
                                  angular_velocity=float(s.angular_velocity + rng.normal(0., spin_std))))
    return replace(board, stones=tuple(stones))


def compare(actual, prediction):
    predicted = {s["id"]: s for s in prediction["stones"]}
    if set(predicted) != {s["id"] for s in actual["stones"]}:
        raise ValueError("Forecast lost or added stone identities.")
    actual_contacts = {tuple(pair) for pair in actual["collisions"]}
    predicted_contacts = {tuple(pair) for pair in prediction["collisions"]}
    rows = []
    for s in actual["stones"]:
        p = predicted[s["id"]]
        rows.append({"id": s["id"], "actual_alive": s["alive"], "predicted_alive": p["alive"],
                     "actual_position": s["position"][:2], "predicted_position": p["position"][:2],
                     "error_m": float(np.linalg.norm(np.asarray(p["position"])[:2] - s["position"][:2]))})
    return {"stones": rows, "worst_stone_error_m": max(s["error_m"] for s in rows),
            "all_within_14cm_and_correct_in_out": all(s["error_m"] <= .14 and s["actual_alive"] == s["predicted_alive"] for s in rows),
            "contact_pairs_match": actual_contacts == predicted_contacts,
            "missed_contacts": [list(p) for p in sorted(actual_contacts - predicted_contacts)],
            "extra_contacts": [list(p) for p in sorted(predicted_contacts - actual_contacts)]}


def summarize(rows, method):
    cases = [r[method] for r in rows]
    stones = [s for result in cases for s in result["stones"]]
    stopped = [s["error_m"] for s in stones if s["actual_alive"]]
    worst = [r["worst_stone_error_m"] for r in cases]
    return {"shots": len(rows), "stones": len(stones), "actual_stopped_stones": len(stopped),
            "actual_out_stones": sum(not s["actual_alive"] for s in stones),
            "mean_stop_error_m": float(np.mean(stopped)), "median_stop_error_m": float(np.median(stopped)),
            "p95_stop_error_m": float(np.quantile(stopped, .95)), "max_stop_error_m": float(np.max(stopped)),
            "mean_worst_stone_error_per_shot_m": float(np.mean(worst)),
            "max_endpoint_error_m": float(np.max(worst)),
            "in_out_accuracy": float(np.mean([s["actual_alive"] == s["predicted_alive"] for s in stones])),
            "all_stones_within_14cm_and_correct_in_out_fraction": float(np.mean([r["all_within_14cm_and_correct_in_out"] for r in cases])),
            "contact_pairs_accuracy": float(np.mean([r["contact_pairs_match"] for r in cases])),
            "missed_contact_pairs": sum(len(r["missed_contacts"]) for r in cases),
            "extra_contact_pairs": sum(len(r["extra_contacts"]) for r in cases)}


def evaluate(regenerate=False):
    REPORT.mkdir(exist_ok=True)
    model = IceRegressor.load(MODEL)
    metadata = json.loads((REPORT / "game_curved_training.json").read_text())
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if digest != metadata["model_sha256"]:
        raise ValueError("Frozen model does not match its training record.")
    used = []
    for split in ("train", "validation", "test"):
        used.extend(json.loads((REPORT / f"game_curved_{split}_shots.json").read_text())["shots"])
    for axis, name in enumerate(("true_friction", "true_curl")):
        if not {pair[axis] for pair in ICE}.isdisjoint({row[name] for row in used}):
            raise ValueError("Collision test must use new ice values.")
    shots = collect(regenerate)
    prior = IceEstimate(model.prior_friction, model.prior_curl)
    rows, summaries, oracle_errors, latencies = [], {}, [], []
    for noise_index, (noise, position_std, velocity_std, spin_std) in enumerate(NOISE):
        rng = np.random.default_rng(SEED + noise_index)
        group = []
        for index, shot in enumerate(shots):
            window = with_noise(window_from(shot), position_std, velocity_std, spin_std, rng)
            clean_board = board_from(shot["board"])
            board = perturb_board(clean_board, window, position_std, velocity_std, spin_std, rng)
            started = time.perf_counter()
            estimate = model.estimate(window)
            learned, _ = predict_board(board, estimate)
            latencies.append(time.perf_counter() - started)
            fixed, _ = predict_board(board, prior)
            if noise == "clean":
                oracle, _ = predict_board(clean_board, IceEstimate(shot["true_friction"], shot["true_curl"]))
                oracle_errors.append(compare(shot["result"], oracle)["worst_stone_error_m"])
            group.append({"id": shot["id"], "case": shot["case"], "noise": noise, "sweep_mode": shot["sweep_mode"],
                          "spin": shot["command"]["spin"], "true_friction": shot["true_friction"], "true_curl": shot["true_curl"],
                          "estimated_friction": estimate.friction, "estimated_curl": estimate.curl,
                          "fixed": compare(shot["result"], fixed), "learned": compare(shot["result"], learned)})
            if (index + 1) % 12 == 0:
                print(f"evaluate {noise}: {index + 1}/{len(shots)}", flush=True)
        summaries[noise] = {"position_noise_std_m": position_std, "velocity_noise_std_m_s": velocity_std,
                            "angular_velocity_noise_std_rad_s": spin_std,
                            "overall": {method: summarize(group, method) for method in ("fixed", "learned")},
                            "by_case": {case: {method: summarize([r for r in group if r["case"] == case], method)
                                                for method in ("fixed", "learned")} for case in CASES}}
        rows.extend(group)
    clean = summaries["clean"]
    contact_counts = {case: {"shots": sum(s["case"] == case for s in shots),
                            "with_contact": sum(s["case"] == case and bool(s["result"]["collisions"]) for s in shots),
                            "with_two_or_more_pairs": sum(s["case"] == case and len(s["result"]["collisions"]) >= 2 for s in shots)} for case in CASES}
    checks = {
        "new_friction_and_curl_values_with_frozen_model": True,
        "all_windows_end_before_any_contact": all(not any(o["contact"] for o in s["observations"]) for s in shots),
        "all_contact_fixture_families_exercised": all(contact_counts[c]["with_contact"] > 0 for c in CASES[1:]),
        "chain_includes_multiple_contacts": contact_counts["chain"]["with_two_or_more_pairs"] > 0,
        "oracle_restoration_under_5mm": max(oracle_errors) < .005,
        "clean_all_case_means_under_14cm": all(g["learned"]["mean_stop_error_m"] < .14 for g in clean["by_case"].values()),
        "clean_all_case_means_improve_over_fixed": all(g["learned"]["mean_stop_error_m"] < g["fixed"]["mean_stop_error_m"] for g in clean["by_case"].values()),
        "all_noise_overall_means_improve_over_fixed": all(s["overall"]["learned"]["mean_stop_error_m"] < s["overall"]["fixed"]["mean_stop_error_m"] for s in summaries.values()),
        "clean_in_out_classification_perfect": clean["overall"]["learned"]["in_out_accuracy"] == 1.,
        "clean_contact_pairs_match": clean["overall"]["learned"]["contact_pairs_accuracy"] == 1.,
        "prediction_under_one_second": max(latencies) < 1.,
    }
    payload = {"status": "passed" if all(checks.values()) else "needs_review", "seed": SEED,
               "model_sha256": digest, "physics_sha256": source_hash(), "mujoco_version": mujoco.__version__,
               "prediction_time_after_release_seconds": .7, "independent_test_shots": len(shots), "conditions": ICE,
               "noise_evaluations": len(rows), "max_oracle_error_m": max(oracle_errors), "max_prediction_seconds": max(latencies),
               "checks": checks, "contact_counts": contact_counts, "summaries": summaries,
               "limitations": ["Same simulator and known contact, curl and sweep laws; not real-ice validation",
                               "Single proposed shot in prepared layouts, not full three-shot adaptive games",
                               "Ice learned from pre-contact motion, not from collision acceleration",
                               "Evaluation is during flight at 0.7 seconds, not learned pre-shot prediction",
                               "No future sweeping; synthetic independent noise on positions/velocities/spin of all live stones",
                               "Robot encoders, game events, stone identities and executed sweep history are exact",
                               "Same 96 trials reused across noise levels; no model tuning on these cases",
                               "Mean stone errors can hide worst-stone errors; both are reported",
                               "Contact labels describe initial layouts; actual misses and contact-pair mismatches remain in evaluation"],
               "rows": rows}
    (REPORT / "game_collision_evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
    flat = []
    for row in rows:
        for method in ("fixed", "learned"):
            for stone in row[method]["stones"]:
                flat.append({"trial": row["id"], "case": row["case"], "noise": row["noise"], "method": method,
                             "stone_id": stone["id"], "actual_alive": stone["actual_alive"], "predicted_alive": stone["predicted_alive"],
                             "error_m": stone["error_m"], "contact_pairs_match": row[method]["contact_pairs_match"]})
    with (REPORT / "game_collision_evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plot(payload)
    print(json.dumps({"status": payload["status"], "checks": checks, "contact_counts": contact_counts,
                      "summary_cm": {name: {case: {"fixed": g["fixed"]["mean_stop_error_m"] * 100,
                                                  "learned": g["learned"]["mean_stop_error_m"] * 100,
                                                  "worst": g["learned"]["max_endpoint_error_m"] * 100,
                                                  "pair_accuracy": g["learned"]["contact_pairs_accuracy"]}
                                            for case, g in summary["by_case"].items()} for name, summary in summaries.items()},
                      "max_prediction_seconds": max(latencies), "max_oracle_error_m": max(oracle_errors)}, indent=2), flush=True)
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
    for axis, noise, title in zip(axes, ("small_noise", "larger_noise"), ("Small observation noise", "Larger observation noise")):
        for offset, (method, label, color) in enumerate((("fixed", "Fixed ice", "#bc7452"), ("learned", "Learned ice", "#23899d"))):
            means = [payload["summaries"][noise]["by_case"][c][method]["mean_stop_error_m"] * 100 for c in CASES]
            bars = axis.bar(np.arange(4) + (offset - .5) * .34, means, width=.34, color=color, label=label)
            axis.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        axis.set_xticks(np.arange(4), ["Free", "Head-on", "Grazing", "Chain"])
        axis.set_ylabel("Mean error per actual in-rink stone (cm)")
        axis.set_title(title)
        axis.legend(fontsize=9)
    fig.savefig(REPORT / "game_collision_evaluation.png", dpi=180)
    plt.close(fig)


def diagnose_failures():
    """Post-evaluation attribution only; never retune the frozen model here."""
    evaluation = json.loads((REPORT / "game_collision_evaluation.json").read_text())
    if evaluation["physics_sha256"] != source_hash() or evaluation["model_sha256"] != hashlib.sha256(MODEL.read_bytes()).hexdigest():
        raise ValueError("Diagnosis must use the model and physics from the saved evaluation.")
    shots = json.loads((REPORT / "game_collision_shots.json").read_text())["shots"]
    noisy_rows = [r for r in evaluation["rows"] if r["noise"] == "larger_noise"]
    worst = max(noisy_rows, key=lambda r: r["learned"]["worst_stone_error_m"])
    selected = {worst["id"]: "largest endpoint error"}
    for row in noisy_rows:
        if not row["learned"]["contact_pairs_match"]:
            selected.setdefault(row["id"], "missed or extra contact")
    _, position_std, velocity_std, spin_std = NOISE[2]
    rng = np.random.default_rng(SEED + 2)
    results = []
    for shot in shots:
        window = with_noise(window_from(shot), position_std, velocity_std, spin_std, rng)
        clean = board_from(shot["board"])
        noisy = perturb_board(clean, window, position_std, velocity_std, spin_std, rng)
        if shot["id"] not in selected:
            continue
        row = next(r for r in noisy_rows if r["id"] == shot["id"])
        learned = IceEstimate(row["estimated_friction"], row["estimated_curl"])
        truth = IceEstimate(shot["true_friction"], shot["true_curl"])
        variants = {}
        for name, observation, ice in (("clean_state_true_ice", clean, truth),
                                       ("clean_state_estimated_ice", clean, learned),
                                       ("noisy_state_true_ice", noisy, truth),
                                       ("noisy_state_estimated_ice", noisy, learned)):
            prediction, _ = predict_board(observation, ice)
            variants[name] = compare(shot["result"], prediction)
        if abs(variants["noisy_state_estimated_ice"]["worst_stone_error_m"] - row["learned"]["worst_stone_error_m"]) > 1e-9:
            raise RuntimeError("Noise replay does not reproduce the selected failure.")
        results.append({"id": shot["id"], "selection_reason": selected[shot["id"]],
                        "true_ice": asdict(truth), "estimated_ice": asdict(learned), "variants": variants})
    payload = {"purpose": "Post-hoc diagnosis of selected failures, not a new accuracy benchmark or model tuning",
               "noise": "larger_noise", "cases": results,
               "note": "Error contributions interact nonlinearly and must not be added together."}
    (REPORT / "game_collision_failure_analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps([{ "id": r["id"], "reason": r["selection_reason"],
                       "variants": {k: {"worst_cm": v["worst_stone_error_m"] * 100,
                                        "contact_pairs_match": v["contact_pairs_match"]} for k, v in r["variants"].items()}}
                      for r in results], indent=2), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "evaluate", "diagnose"))
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    if args.command == "diagnose":
        diagnose_failures()
        return
    if evaluate(args.regenerate)["status"] != "passed":
        raise SystemExit("Collision gate needs review; see the saved evaluation.")


if __name__ == "__main__":
    main()
