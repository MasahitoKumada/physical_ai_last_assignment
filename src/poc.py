"""Technical PoC: curling robot, collisions, reference forecasts, and a native UI.

The forecast uses explicitly known ice parameters. This is a consistency and
latency test; it is not a trained AI predictor or the final submission.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from physics import COLORS, CurlingWorld, DT, HOUSE_RADIUS, RADIUS, ShotCommand, reference_forecast, score_stone

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
VIDEO = ROOT / "video"


def fixture(name):
    world = CurlingWorld()
    command = ShotCommand(0, 79, 1)
    if name == "right_spin":
        command = ShotCommand(0, 79, -1)
    elif name == "head_on":
        world.place_fixture(1, (2., 0.))
        command = ShotCommand(0, 82, 0)
    elif name == "grazing":
        world.place_fixture(1, (2., .10))
        command = ShotCommand(0, 82, 0)
    elif name == "chain":
        world.place_fixture(1, (2., 0.))
        world.place_fixture(2, (2.3, 0.))
        command = ShotCommand(0, 88, 0)
    elif name not in ("left_spin", "sweep"):
        raise ValueError(name)
    return world, command


def verify():
    started = time.perf_counter()
    REPORT.mkdir(exist_ok=True)
    checks = []
    cases = []
    for name in ("left_spin", "right_spin", "head_on", "grazing", "chain"):
        world, command = fixture(name)
        snapshot = world.snapshot()
        before_fields = json.dumps(world.result(), sort_keys=True)
        t0 = time.perf_counter()
        predicted, _ = reference_forecast(snapshot, command, .05, .03)
        elapsed = time.perf_counter() - t0
        assert np.array_equal(snapshot["state"], world.snapshot()["state"]), "Forecast changed live integration state"
        assert before_fields == json.dumps(world.result(), sort_keys=True), "Forecast changed live game state"
        world.start_shot(command)
        actual = world.run_until_rest()
        errors = [float(np.linalg.norm(np.array(a["position"]) - b["position"]))
                  for a, b in zip(actual["stones"], predicted["stones"])]
        assert max(errors, default=0.) < 1e-6
        assert elapsed < 1., f"Reference forecast too slow: {elapsed} seconds"
        expected = {"head_on": {(0, 1)}, "grazing": {(0, 1)}, "chain": {(0, 1), (1, 2)}}.get(name, set())
        assert expected.issubset(set(map(tuple, actual["collisions"])))
        if command.spin:
            assert actual["release"]["velocity"][2] * command.spin > .5
        release_speed = np.linalg.norm(actual["release"]["velocity"][3:5])
        assert .7 * command.speed < release_speed < 1.3 * command.speed
        assert actual["max_passive_energy"] <= actual["release"]["stone_kinetic_energy"] * 1.05 + 1e-6
        cases.append({"name": name, "command": asdict(command), "reference_forecast_seconds": elapsed,
                      "same_ice_position_errors_m": errors, "actual": actual})
    checks += ["Actuator-driven release has commanded translation and spin",
               "Head-on, grazing, and three-stone chain contacts occur",
               "Reference prediction preserves live physical and game state",
               "Known-ice forecast matches the independently executed shot",
               "Reference forecast finishes within one second",
               "Passive stone kinetic energy stays below 105% of release energy"]
    left, right = cases[0]["actual"]["stones"][0]["position"], cases[1]["actual"]["stones"][0]["position"]
    assert left[1] > .01 and right[1] < -.01
    assert abs(left[0] - right[0]) < .01
    checks.append("Reversing spin reverses the lateral curl")

    plain_world, command = fixture("left_spin")
    plain_world.start_shot(command)
    plain = plain_world.run_until_rest()
    swept_world, command = fixture("sweep")
    swept_world.start_shot(command)
    swept = swept_world.run_until_rest(sweeping=True)
    assert swept["stones"][0]["position"][0] > plain["stones"][0]["position"][0] + .1
    sweep_result = {"without_sweep": plain, "with_sweep": swept}
    checks.append("Sweeping increases travel distance with identical initial state and input")

    # Variation exercises motor speed, rotation, friction, curl, and removal together.
    rng = np.random.default_rng(20261030)
    stability = []
    for index in range(20):
        world = CurlingWorld(friction=float(rng.uniform(.035, .065)))
        command = ShotCommand(float(rng.uniform(-12, 12)), float(rng.uniform(35, 95)), int(rng.integers(-1, 2)))
        world.start_shot(command)
        result = world.run_until_rest(sweeping=bool(index % 2))
        assert result["max_height"] < .10 and result["max_speed"] < 4.
        stability.append({"command": asdict(command), "max_height": result["max_height"], "phase": result["phase"]})
    checks.append("20 varied shots complete without invalid state, excessive lift, or timeout")

    # Retry exactly reproduces the same shot, including the reset of the three-stone state.
    world = CurlingWorld()
    world.start_shot(ShotCommand(0, 79, 1))
    first = world.run_until_rest()
    world.reset()
    world.start_shot(ShotCommand(0, 79, 1))
    second = world.run_until_rest()
    assert first == second
    checks.append("Reset and replay reproduce identical results")
    for angle in (-14., 0., 14.):
        command = ShotCommand(angle, 75, 0)
        world = CurlingWorld()
        world.start_shot(command)
        while world.release is None:
            world.step()
        # A flight-state forecast should preserve integration state too.
        for _ in range(150):
            world.step()
        snapshot = world.snapshot()
        predicted, _ = reference_forecast(snapshot, command, .05, .03)
        assert np.array_equal(snapshot["state"], world.snapshot()["state"])
        actual = world.run_until_rest()
        assert np.allclose(predicted["stones"][0]["position"], actual["stones"][0]["position"], atol=1e-6)
    checks.append("Snapshots taken during flight reproduce final positions")

    scores = [score_stone((3 + d, 0)) for d in np.linspace(0, HOUSE_RADIUS + RADIUS + .01, 1001)]
    assert scores[0] == 100 and scores[-1] == 0
    assert all(a >= b for a, b in zip(scores, scores[1:]))
    assert score_stone((3 + HOUSE_RADIUS - RADIUS, 0)) == 1
    assert score_stone((3 + HOUSE_RADIUS + RADIUS, 0)) == 1
    assert score_stone((3, 0), active=False) == 0
    checks.append("Distance scores are monotonic and house boundary stones receive one point")

    game = CurlingWorld()
    game_results = []
    for number, command in enumerate([ShotCommand(0, 79, 0), ShotCommand(2, 78, -1), ShotCommand(-2, 80, 1)]):
        game.friction = .05 * (1 - .05 * number)
        game.start_shot(command)
        game_results.append(game.run_until_rest())
    assert game.phase == "RESULT" and game.played.sum() == 3
    try:
        game.start_shot(ShotCommand())
    except ValueError:
        pass
    else:
        raise AssertionError("Fourth shot accepted")
    checks.append("Three-shot game with changing ice finishes, retains stones, and rejects a fourth shot")

    # Sensitivity is a diagnostic, not a learned-model accuracy measurement.
    sensitivity = []
    for name in ("head_on", "grazing", "chain"):
        world, command = fixture(name)
        snapshot = world.snapshot()
        forecast, _ = reference_forecast(snapshot, command, .0475, .03)
        world.start_shot(command)
        actual = world.run_until_rest()
        sensitivity.append({"case": name, "assumed_friction": .0475, "true_friction": .05,
                            "errors_m": [float(np.linalg.norm(np.array(a["position"])[:2] - b["position"][:2]))
                                         for a, b in zip(actual["stones"], forecast["stones"])]})
    for name in ("grazing", "chain"):
        base_world, command = fixture(name)
        snapshot = base_world.snapshot()
        base, _ = reference_forecast(snapshot, command, .05, .03)
        for mu in (.049999, .050001):
            perturbed, _ = reference_forecast(snapshot, command, mu, .03)
            errors = [np.linalg.norm(np.asarray(a["position"]) - b["position"])
                      for a, b in zip(base["stones"], perturbed["stones"])]
            assert max(errors) < .005, f"Excessive contact sensitivity: {name}: {errors}"
    checks.append("Microscopic friction changes produce less than 5 mm endpoint change in diagnostic contacts")
    result = {"status": "passed", "purpose": "Technical feasibility, not learned AI accuracy",
              "mujoco_version": mujoco.__version__, "motion_model": "planar circles with explicit Coulomb drag",
              "checks": checks,
              "reference_cases": cases, "sweep_comparison": sweep_result,
              "stability_trials": stability, "three_shot_game": game_results,
              "friction_sensitivity": sensitivity,
              "elapsed_seconds": time.perf_counter() - started}
    (REPORT / "poc_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks_passed": len(checks),
                      "slowest_reference_forecast_seconds": max(c["reference_forecast_seconds"] for c in cases),
                      "three_shot_score": game_results[-1]["score"],
                      "friction_sensitivity": sensitivity, "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


def compare_contact():
    records = []
    for name, planar, solref in [("3d_stiff", False, (.006, .25)), ("3d_soft", False, (.02, .25)),
                                ("planar_soft", True, (.02, .25))]:
        for case in ("grazing", "chain"):
            source, command = fixture(case)
            positions = [(s, source.positions()[s, :2]) for s in np.flatnonzero(source.played)]
            outcomes = []
            for mu in (.05, .050001, .049999, .0475):
                world = CurlingWorld(mu, planar=planar)
                for j, (a, b) in enumerate(zip(world.model.pair_geom1, world.model.pair_geom2)):
                    if a in world.disk_ids and b in world.disk_ids:
                        world.model.pair_solref[j] = solref
                for index, xy in positions:
                    world.place_fixture(int(index), xy)
                world.start_shot(command)
                outcomes.append({"friction": mu, "result": world.run_until_rest()})
            p = np.array([[stone["position"] for stone in outcome["result"]["stones"]] for outcome in outcomes])
            records.append({"model": name, "case": case, "collision_solref": solref,
                            "micro_max_delta_m": float(np.max(np.linalg.norm(p[1:3, :, :2] - p[0, :, :2], axis=2))),
                            "five_percent_max_delta_m": float(np.max(np.linalg.norm(p[3, :, :2] - p[0, :, :2], axis=1))),
                            "outcomes": outcomes})
    REPORT.mkdir(exist_ok=True)
    (REPORT / "poc_contact_comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for offset, (model, label, color) in enumerate([
            ("3d_stiff", "3D / initial contact", "#b66b4c"),
            ("3d_soft", "3D / adjusted contact", "#cda75a"),
            ("planar_soft", "Planar / adjusted contact", "#23899d")]):
        values = [next(r["micro_max_delta_m"] * 1000 for r in records if r["model"] == model and r["case"] == case)
                  for case in ("grazing", "chain")]
        axis.bar(np.arange(2) + (offset - 1) * .24, values, width=.24, label=label, color=color)
    axis.set_xticks([0, 1], ["Grazing collision", "Three-stone chain"])
    axis.set_yscale("log")
    axis.set_ylabel("Maximum endpoint change (mm, log scale)")
    axis.set_title("Diagnostic: friction 0.05 perturbed by +/- 0.000001")
    axis.legend(fontsize=9)
    fig.savefig(REPORT / "poc_contact_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps([{k: v for k, v in row.items() if k != "outcomes"} for row in records], indent=2))


class PocRenderer:
    def __init__(self, world, width=960, height=540):
        self.world = world
        self.renderer = mujoco.Renderer(world.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [2., 0., .06]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 4.8, 90, -57
        self.prediction = None
        self.predicted_trace = []

    def _geom(self, kind, size, position, color, matrix=None):
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.asarray(size), np.asarray(position),
                           np.eye(3).ravel() if matrix is None else matrix.ravel(), np.asarray(color))
        scene.ngeom += 1
        return geom

    def render(self):
        world = self.world
        self.renderer.update_scene(world.data, self.camera)
        if self.prediction:
            for stone in self.prediction["stones"]:
                if stone["alive"]:
                    color = list(COLORS[stone["id"]])
                    color[3] = .35
                    p = stone["position"]
                    self._geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [.073, .028, 0.], [p[0], p[1], .028], color)
            for index in range(3):
                previous = None
                for point in self.predicted_trace[::3]:
                    p = point[index]
                    if 0 <= p[0] <= 4 and abs(p[1]) <= .75:
                        if previous is not None and np.linalg.norm(np.asarray(p[:2]) - previous) < .045:
                            continue
                        color = list(COLORS[index])
                        self._geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.005] * 3, [p[0], p[1], .008], color)
                        previous = np.asarray(p[:2])
        if world.sweeping and world.current is not None and world.alive[world.current]:
            v = world.velocities()[world.current, 3:5]
            speed = np.linalg.norm(v)
            if speed > .01:
                direction = v / speed
                position = world.positions()[world.current].copy()
                position[:2] += direction * .15
                position[2] = .025
                rotation = np.array([[direction[0], -direction[1], 0], [direction[1], direction[0], 0], [0, 0, 1.]])
                self._geom(mujoco.mjtGeom.mjGEOM_BOX, [.025, .09, .012], position, [.25, .85, .4, 1.], rotation)
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


def font_path():
    for path in ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "C:/Windows/Fonts/meiryo.ttc"):
        if Path(path).is_file():
            return path
    raise FileNotFoundError("Install a Japanese font to render the PoC video.")


def video():
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    VIDEO.mkdir(exist_ok=True)
    REPORT.mkdir(exist_ok=True)
    fonts = {size: ImageFont.truetype(font_path(), size) for size in (20, 25, 32, 40)}
    labels = [("left_spin", "左回転：ロボットの保持・回転・解放"),
              ("right_spin", "右回転：曲がる方向の反転"),
              ("head_on", "正面衝突：先に置いた石を押す"),
              ("grazing", "かすり衝突：横方向への分岐"),
              ("chain", "3個の連鎖衝突"),
              ("sweep", "スイープ：到達距離の変化")]
    frames = 0
    with imageio.get_writer(VIDEO / "game_poc.mp4", fps=25, codec="libx264", quality=8,
                           macro_block_size=16, ffmpeg_log_level="error") as writer:
        title = Image.new("RGB", (1280, 720), "#101d2e")
        draw = ImageDraw.Draw(title)
        draw.text((60, 100), "3投カーリング：技術検証PoC", font=fonts[40], fill="white")
        for i, line in enumerate(["回転付き投球・衝突・先読みを確認", "半透明の石と点線：既知の氷条件での物理予測",
                                  "このPoCには学習済みAIはまだ接続していません。"]):
            draw.text((65, 240 + 65 * i), line, font=fonts[25], fill="#bed5e6")
        for _ in range(75):
            writer.append_data(np.asarray(title))
            frames += 1
        for name, label in labels:
            world, command = fixture(name)
            prediction, trace = reference_forecast(world.snapshot(), command, .05, .03)
            renderer = PocRenderer(world, 1280, 576)
            renderer.prediction, renderer.predicted_trace = prediction, trace
            world.start_shot(command)
            def capture(w):
                nonlocal frames
                canvas = Image.new("RGB", (1280, 720), "#101d2e")
                canvas.paste(Image.fromarray(renderer.render()), (0, 88))
                d = ImageDraw.Draw(canvas)
                d.text((30, 13), label, font=fonts[32], fill="white")
                d.text((32, 55), f"方向 {command.angle:+.0f}° / パワー {command.power:.0f} / 再生：等速", font=fonts[20], fill="#abc7dd")
                caption = "点線はスイープなしの予測。手動スイープで結果が変わります。" if name == "sweep" else "PoC：既知条件による物理予測。AIの予測精度評価は未実施。"
                d.text((30, 676), caption, font=fonts[20], fill="#d3e5f2")
                writer.append_data(np.asarray(canvas))
                frames += 1
            result = world.run_until_rest(sweeping=name == "sweep", callback=capture)
            for _ in range(20):
                capture(world)
            if name == "chain":
                imageio.imwrite(REPORT / "poc_preview.png", renderer.render())
            renderer.close()
        ending = title.copy()
        d = ImageDraw.Draw(ending)
        d.rectangle((0, 0, 1280, 720), fill="#101d2e")
        d.text((60, 100), "次の検証：未知の氷を学習して予測", font=fonts[40], fill="white")
        d.text((65, 250), "技術検証用の映像です。最終提出用動画ではありません。", font=fonts[25], fill="#bed5e6")
        for _ in range(75):
            writer.append_data(np.asarray(ending))
            frames += 1
    metadata = {"path": "video/game_poc.mp4", "fps": 25, "frames": frames, "duration_seconds": frames / 25,
                "learned_ai": False, "prediction": "known-ice reference rollout", "submission_video": False}
    (REPORT / "poc_video_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "compare-contact", "video", "play", "gui-smoke"])
    args = parser.parse_args()
    if args.command == "check":
        verify()
    elif args.command == "compare-contact":
        compare_contact()
    elif args.command == "video":
        video()
    else:
        from ui import run_poc_ui
        run_poc_ui(PocRenderer, smoke_seconds=6. if args.command == "gui-smoke" else None)


if __name__ == "__main__":
    main()
