"""Joint-actuated robot curling: train, evaluate, render, and prepare a report.

The policy receives a probe observation, never the environment's friction value.
Puck velocities are produced by contact, not assigned by the controller.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
VIDEO = ROOT / "video"
TARGET_X = 1.2
SUCCESS_RADIUS = .15
PROBE_SPEED = .65
SPEED_MIN, SPEED_MAX = .35, 1.5
TRAIN_MU = (.025, .04, .055, .07, .085, .10, .115, .13)
TEST_MU = tuple(.028 + .005 * i for i in range(20)) + (.127,)
DEMO_MU = (.033, .078, .123)
DT, DURATION, FPS = .002, 8., 25
STROKE = .26
SEED = 20261027


def make_scene(friction: float) -> mujoco.MjModel:
    """Friction is environment configuration, separate from the learned policy."""
    if not .02 <= friction <= .15:
        raise ValueError("This experiment supports friction in [0.02, 0.15].")
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("scene.xml")))
    pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, "sliding")
    model.pair_friction[pair, :2] = friction
    return model


def rollout(model: mujoco.MjModel, speed: float,
            frame_callback: Callable | None = None) -> dict:
    """Run one shot from a fresh initial state; control only the two robot motors."""
    if not SPEED_MIN <= speed <= SPEED_MAX:
        raise ValueError(f"Speed must be between {SPEED_MIN} and {SPEED_MAX}.")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    puck = model.body("puck").id
    free_joint = model.joint("puck_free").id
    velocity_address = model.jnt_dofadr[free_joint]
    pad, disk = model.geom("pad").id, model.geom("disk").id
    release_time = STROKE / speed
    peak_v, max_z, contact_steps, post_release_contact_steps = 0., 0., 0, 0
    sample_interval = round(1 / FPS / DT)
    for step in range(round(DURATION / DT)):
        # Horizontal motion followed by lifting the pad away from the puck.
        data.ctrl[0] = min(speed * data.time, STROKE)
        data.ctrl[1] = min(max(data.time - release_time, 0.) * 2., .20)
        mujoco.mj_step(model, data)
        peak_v = max(peak_v, float(abs(data.qvel[velocity_address])))
        max_z = max(max_z, float(data.xpos[puck, 2]))
        touching = any({int(c.geom1), int(c.geom2)} == {pad, disk} for c in data.contact)
        contact_steps += int(touching)
        if data.time > release_time + .3:
            post_release_contact_steps += int(touching)
        if frame_callback is not None and step % sample_interval == 0:
            frame_callback(model, data)
    position = data.xpos[puck].copy()
    final_speed = float(np.linalg.norm(data.qvel[velocity_address:velocity_address + 3]))
    if (not np.isfinite(position).all() or not .2 < position[0] < 10
            or max_z > .06 or peak_v > 3. or final_speed > .005
            or contact_steps == 0 or post_release_contact_steps):
        raise RuntimeError(f"Unstable or unfinished shot: speed={speed}, "
                           f"x={position[0]}, z_max={max_z}, v_final={final_speed}, "
                           f"contacts={contact_steps}, late_contacts={post_release_contact_steps}")
    error = float(np.hypot(position[0] - TARGET_X, position[1]))
    return {"x": float(position[0]), "y": float(position[1]), "error_m": error,
            "success": bool(error <= SUCCESS_RADIUS), "speed_command": float(speed),
            "final_speed": final_speed, "peak_vx": peak_v, "max_z": max_z,
            "contact_steps": contact_steps,
            "post_release_contact_steps": post_release_contact_steps}


def features(speed, probe_x):
    v, p = np.broadcast_arrays(np.asarray(speed), np.asarray(probe_x))
    return np.stack([np.ones_like(v), v, v * v, p, v * p, v * v * p], axis=-1)


@dataclass
class ShotPredictor:
    """Small ridge regression with an observed probe position as context."""
    coefficients: np.ndarray

    @classmethod
    def fit(cls, rows: list[dict]):
        x = np.array([features(r["speed_command"], r["probe_x"]) for r in rows])
        y = np.array([r["x"] for r in rows])
        penalty = np.eye(x.shape[1]) * 1e-6
        penalty[0, 0] = 0
        return cls(np.linalg.solve(x.T @ x + penalty, x.T @ y))

    def choose_speed(self, observed_probe_x: float) -> float:
        if not np.isfinite(observed_probe_x):
            raise ValueError("Probe observation must be finite.")
        candidates = np.linspace(SPEED_MIN, SPEED_MAX, 601)
        prediction = features(candidates, observed_probe_x) @ self.coefficients
        return float(candidates[np.argmin(abs(prediction - TARGET_X))])

    def save(self, path: Path):
        payload = {"schema_version": 1, "features": ["1", "v", "v^2", "probe_x", "v*probe_x", "v^2*probe_x"],
                   "coefficients": self.coefficients.tolist(), "target_x": TARGET_X,
                   "probe_speed": PROBE_SPEED, "speed_range": [SPEED_MIN, SPEED_MAX]}
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(path.read_text())
        if (payload["target_x"] != TARGET_X or payload["probe_speed"] != PROBE_SPEED
                or payload["speed_range"] != [SPEED_MIN, SPEED_MAX]):
            raise ValueError("Model settings do not match this experiment; run train again.")
        return cls(np.array(payload["coefficients"]))


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train():
    REPORT.mkdir(exist_ok=True)
    rows = []
    for mu in TRAIN_MU:
        model = make_scene(mu)
        probe_x = rollout(model, PROBE_SPEED)["x"]
        for speed in np.linspace(SPEED_MIN, SPEED_MAX, 12):
            rows.append({"friction": mu, "probe_x": probe_x, **rollout(model, float(speed))})
    predictor = ShotPredictor.fit(rows)
    predictor.save(REPORT / "model.json")
    write_csv(REPORT / "training.csv", rows)
    rmse = np.sqrt(np.mean([(features(r["speed_command"], r["probe_x"]) @ predictor.coefficients - r["x"]) ** 2
                           for r in rows]))
    print(json.dumps({"training_shots": len(rows), "training_rmse_m": float(rmse)}, indent=2))


def calibrate_fixed_speed() -> float:
    """A strong fixed baseline: calibrate once on the nominal floor, then freeze it."""
    nominal = make_scene(.06)
    coarse = np.linspace(SPEED_MIN, SPEED_MAX, 21)
    best = min(coarse, key=lambda v: rollout(nominal, float(v))["error_m"])
    fine = np.linspace(max(SPEED_MIN, best - .06), min(SPEED_MAX, best + .06), 13)
    return float(min(fine, key=lambda v: rollout(nominal, float(v))["error_m"]))


def evaluate():
    predictor = ShotPredictor.load(REPORT / "model.json")
    if any(abs(a - b) < 1e-10 for a in TRAIN_MU for b in TEST_MU):
        raise RuntimeError("Training and test friction conditions must be disjoint.")
    fixed_speed = calibrate_fixed_speed()
    rows = []
    rng = np.random.default_rng(SEED)
    for mu in TEST_MU:
        model = make_scene(mu)
        probe_x = rollout(model, PROBE_SPEED)["x"]
        fixed = rollout(model, fixed_speed)
        adaptive = rollout(model, predictor.choose_speed(probe_x))
        rows.append({"friction": mu, "probe_x": probe_x,
                     "fixed_speed": fixed_speed, "adaptive_speed": adaptive["speed_command"],
                     "fixed_x": fixed["x"], "adaptive_x": adaptive["x"],
                     "fixed_error_m": fixed["error_m"], "adaptive_error_m": adaptive["error_m"],
                     "fixed_success": fixed["success"], "adaptive_success": adaptive["success"]})
    noisy = []
    # Only the observed probe position is perturbed; the simulator remains deterministic.
    for std in (.005, .01):
        errors, successes = [], []
        for row in rows:
            model = make_scene(row["friction"])
            for _ in range(3):
                observation = row["probe_x"] + rng.normal(0., std)
                shot = rollout(model, predictor.choose_speed(float(observation)))
                errors.append(shot["error_m"])
                successes.append(shot["success"])
        noisy.append({"observation_noise_std_m": std, "shots": len(errors),
                      "mean_error_m": float(np.mean(errors)), "success_rate": float(np.mean(successes))})
    summary = {"mujoco_version": mujoco.__version__, "seed": SEED,
               "train_friction": list(TRAIN_MU), "training_shots": len(TRAIN_MU) * 12,
               "test_conditions": len(rows), "target_x_m": TARGET_X,
               "success_radius_m": SUCCESS_RADIUS, "fixed_speed": fixed_speed,
               "mean_fixed_error_m": float(np.mean([r["fixed_error_m"] for r in rows])),
               "mean_adaptive_error_m": float(np.mean([r["adaptive_error_m"] for r in rows])),
               "fixed_success_rate": float(np.mean([r["fixed_success"] for r in rows])),
               "adaptive_success_rate": float(np.mean([r["adaptive_success"] for r in rows])),
               "noise_checks": noisy,
               "limitations": ["Straight shots only", "Uniform Coulomb contact friction",
                               "Simulator ground-truth position observation", "Fixed puck mass and geometry",
                               "Held-out friction is inside the training range", "No real-world validation"],
               "conditions": rows}
    (REPORT / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(REPORT / "evaluation.csv", rows)
    plot_evaluation(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, indent=2))


def plot_evaluation(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    rows = summary["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    x = [r["friction"] for r in rows]
    axes[0].plot(x, [r["fixed_x"] for r in rows], "o-", color="#b96b41", label="Fixed speed")
    axes[0].plot(x, [r["adaptive_x"] for r in rows], "o-", color="#197f99", label="Probe + learned model")
    axes[0].axhspan(TARGET_X - SUCCESS_RADIUS, TARGET_X + SUCCESS_RADIUS, color="#cde6d4", alpha=.6)
    axes[0].axhline(TARGET_X, color="#42566d", ls="--", label="Target")
    axes[0].set(xlabel="Floor friction (evaluation label only)", ylabel="Final puck position (m)")
    axes[0].legend(fontsize=8)
    axes[1].bar(["Fixed speed", "Probe + model"],
                [summary["mean_fixed_error_m"] * 100, summary["mean_adaptive_error_m"] * 100],
                color=["#b96b41", "#197f99"])
    axes[1].set(ylabel="Mean target error (cm)", title=f"{len(rows)} held-out friction conditions")
    fig.savefig(REPORT / "evaluation.png", dpi=180)
    plt.close(fig)


def find_font(explicit: str | None = None) -> Path:
    candidates = [explicit] if explicit else [
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/meiryo.ttc"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Japanese font not found; pass --font /path/to/font.ttc")


def render_video(font_argument: str | None = None):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    predictor = ShotPredictor.load(REPORT / "model.json")
    summary = json.loads((REPORT / "evaluation.json").read_text())
    font_path = find_font(font_argument)
    font = lambda size: ImageFont.truetype(str(font_path), size)
    f20, f25, f32, f40 = (font(s) for s in (20, 25, 32, 40))
    VIDEO.mkdir(exist_ok=True)
    path = VIDEO / "curling_demo.mp4"
    frame_count = 0
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                           macro_block_size=16, ffmpeg_log_level="error") as writer:
        def append(frame):
            nonlocal frame_count
            writer.append_data(np.asarray(frame))
            frame_count += 1

        def card(title, lines, seconds):
            canvas = Image.new("RGB", (1280, 720), "#101d2e")
            draw = ImageDraw.Draw(canvas)
            draw.text((70, 95), title, font=f40, fill="#ffffff")
            for i, line in enumerate(lines):
                draw.text((75, 215 + i * 68), line, font=f25, fill="#c9dceb")
            draw.text((75, 650), "MuJoCo / 直交ロボット / 回帰モデルによる操作予測", font=f20, fill="#8da7bf")
            for _ in range(round(seconds * FPS)):
                append(canvas)

        card("ロボット・カーリング", ["目標：円盤を押し出し、的の中心で止める。",
             "床の滑りやすさが変わると、同じ押し方では届かない。",
             "試し打ちの観測から、AIが次の押す速さを選ぶ。",
             "シミュレーション時間と同じ速さで再生"], 6)
        demo_results = []
        for index, (mu, label) in enumerate(zip(DEMO_MU, ["滑りやすい床", "中間の床", "滑りにくい床"]), 1):
            model = make_scene(mu)
            probe = rollout(model, PROBE_SPEED)
            adaptive_speed = predictor.choose_speed(probe["x"])
            phases = [("固定した押し方", summary["fixed_speed"], "基準の床で調整した速さを、そのまま使う。"),
                      ("試し打ち", PROBE_SPEED, "円盤の停止位置を観測する。摩擦係数の正解はAIに渡さない。"),
                      ("AIによる調整", adaptive_speed, "学習済みの回帰モデルが、停止位置を予測して速さを選ぶ。")]
            card(f"条件 {index} / {label}", ["固定操作 → 試し打ち → AIによる調整", "同じ円盤・同じ的で比較する。"], 2)
            with mujoco.Renderer(model, height=576, width=1280) as renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [1.35, 0, .08]
                camera.distance, camera.azimuth, camera.elevation = 3.5, 90, -53
                results = []
                for phase, speed, caption in phases:
                    trail = []
                    def draw_frame(m, data):
                        renderer.update_scene(data, camera)
                        pos = data.xpos[m.body("puck").id].copy()
                        if not trail or np.linalg.norm(pos[:2] - trail[-1][:2]) > .025:
                            trail.append(pos)
                        for point in trail:
                            scene = renderer.scene
                            if scene.ngeom >= scene.maxgeom:
                                break
                            geom = scene.geoms[scene.ngeom]
                            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                               np.array([.006, .006, .006]),
                                               np.array([point[0], point[1], .008]),
                                               np.eye(3).ravel(), np.array([.05, .65, .75, 1.]))
                            scene.ngeom += 1
                        canvas = Image.new("RGB", (1280, 720), "#101d2e")
                        canvas.paste(Image.fromarray(renderer.render()), (0, 88))
                        draw = ImageDraw.Draw(canvas)
                        draw.text((30, 15), f"条件 {index}：{label}  |  {phase}", font=f32, fill="white")
                        draw.text((32, 56), f"押す速さ {speed:.2f} m/s   |   再生：等速", font=f20, fill="#a9c5dc")
                        error_cm = np.linalg.norm(pos[:2] - [TARGET_X, 0.]) * 100
                        draw.text((895, 104), f"的まで {error_cm:5.1f} cm", font=f25, fill="white", stroke_width=1, stroke_fill="#101d2e")
                        draw.text((30, 676), caption, font=f20, fill="#d7e8f5")
                        append(canvas)
                    result = rollout(model, float(speed), draw_frame)
                    results.append(result)
                demo_results.append({"label": label, "friction": mu,
                                     "fixed_error_m": results[0]["error_m"],
                                     "probe_x": probe["x"], "adaptive_error_m": results[2]["error_m"]})
        fixed_cm = summary["mean_fixed_error_m"] * 100
        adaptive_cm = summary["mean_adaptive_error_m"] * 100
        card("未知の床で、押し方を変えられたか？", [
             f"学習とは別の摩擦条件 {summary['test_conditions']} 種類で評価",
             f"平均誤差：固定操作 {fixed_cm:.1f} cm → AI調整 {adaptive_cm:.1f} cm",
             f"的の中心から15 cm以内：{summary['adaptive_success_rate']:.0%}",
             "範囲：直線ショット・位置はシミュレータから取得"], 8)
    metadata = {"path": "video/curling_demo.mp4", "fps": FPS,
                "frames": frame_count, "duration_seconds": frame_count / FPS,
                "playback_speed": 1, "demo_conditions": demo_results}
    (REPORT / "video_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if not 60 <= frame_count / FPS <= 180:
        raise RuntimeError("Submission video must be between 60 and 180 seconds.")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def make_report(font_argument: str | None = None):
    """Generate a reviewable draft PDF; account-specific naming is a submission step."""
    from PIL import Image, ImageDraw, ImageFont
    summary = json.loads((REPORT / "evaluation.json").read_text())
    font_path = find_font(font_argument)
    fixed = summary["mean_fixed_error_m"] * 100
    adaptive = summary["mean_adaptive_error_m"] * 100
    pages = [
        ("ロボット・カーリング", [
            "1. 概要・目的",
            "床の滑りやすさが分からない状況で、直交ロボットが円盤を押し出し、的の中心で止めるタスクを実装した。物理学のモデル化と、観測から操作を予測する機械学習を組み合わせる。",
            "2. ロボットとシミュレーション",
            "MuJoCoを使用。水平移動と上下移動の2軸をモーターで駆動する直交ロボットを作成した。押し子の接触で円盤を加速し、その後は押し子を上げ、円盤が摩擦で停止する。円盤へ直接速度を設定する制御は行わない。",
            "3. AIの役割",
            "試し打ちの停止位置と操作速度を入力とし、円盤の最終位置を予測する回帰モデルを学習した。候補となる速度ごとに停止位置を予測し、的に最も近い予測となる速度を選択する。AIには摩擦係数の正解を入力しない。",
            f"学習データは摩擦8条件×速度12条件の{summary['training_shots']}ショット。入力特徴は1、v、v²、試し打ちの停止位置p、vp、v²p。L2正則化付き最小二乗法で学習した。",
            "4. 工夫した点",
            "同じ形・質量の円盤を使い、床の摩擦だけを変えることで、試し打ちの意義を可視化した。比較対象は基準の床で一度調整し、その後は固定した操作速度とした。レール・可動部・的・軌跡・距離を表示し、動画だけでも目的と結果を追えるようにした。",
        ]),
        ("評価・考察", [
            "5. 評価方法",
            f"学習に使用していない{summary['test_conditions']}種類の摩擦条件で比較した。評価条件は学習範囲内に置いた。主指標は的の中心からの最終距離で、15 cm以内を成功とした。円盤は毎回同じ位置・静止状態から開始する。",
            f"固定操作の平均誤差：{fixed:.2f} cm。試し打ち＋学習モデルの平均誤差：{adaptive:.2f} cm。",
            f"成功率：固定操作 {summary['fixed_success_rate']:.1%}、AI調整 {summary['adaptive_success_rate']:.1%}。",
            "6. 結果の解釈と限界",
            "本結果は一様な接触摩擦、固定の円盤質量・形状、直線ショットに限定される。AIへの入力はシミュレータから取得した位置で、カメラによる認識ではない。未知の摩擦条件への対応は学習範囲内の補間であり、範囲外への汎化や実機への転用は検証していない。",
            "観測ノイズの検証では、試し打ちの停止位置だけにガウスノイズを加えた。接触・アクチュエータ・全センサーに現実の誤差を加えた検証ではない。詳細な数値はevaluation.jsonに保存した。",
            "7. 今後の拡張",
            "直交ロボット版を主成果とし、時間に余裕があれば多関節アームを比較対象として追加する。円盤・床・的・観測・評価指標を揃える。ただし回帰モデルは各ロボットの実際の操作特性に合わせて再学習する。",
            "8. 再現と提出物",
            "READMEに環境構築と実行手順を記載。学習データ、学習済み係数、評価結果、評価図をreport/に保存する。動画はvideo/curling_demo.mp4。公開GitHubのURL、説明資料PDF、YouTube動画URLを提出する。",
        ])]
    markdown = "# ロボット・カーリング\n\n"
    images = []
    for page_index, (title, paragraphs) in enumerate(pages):
        image = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(image)
        title_font = ImageFont.truetype(str(font_path), 42)
        body_font = ImageFont.truetype(str(font_path), 25)
        heading_font = ImageFont.truetype(str(font_path), 29)
        draw.text((85, 70), title, font=title_font, fill="#193653")
        y = 155
        for paragraph in paragraphs:
            is_heading = len(paragraph) < 35 and paragraph[:1].isdigit()
            selected_font = heading_font if is_heading else body_font
            line = ""
            for char in paragraph:
                if draw.textlength(line + char, font=selected_font) > 1070:
                    draw.text((85, y), line, font=selected_font, fill="#26384a")
                    y += 39
                    line = ""
                line += char
            draw.text((85, y), line, font=selected_font, fill="#26384a")
            y += 62 if is_heading else 64
            markdown += ("## " if is_heading else "") + paragraph + "\n\n"
        if y > 1660:
            raise RuntimeError("Report text overflow: shorten paragraphs or add a page.")
        draw.text((1100, 1680), str(page_index + 1), font=body_font, fill="#708095")
        images.append(image)
    # Add the standalone evaluation plot as a third page.
    chart_page = Image.new("RGB", (1240, 1754), "white")
    chart_draw = ImageDraw.Draw(chart_page)
    chart_draw.text((85, 70), "評価図", font=ImageFont.truetype(str(font_path), 42), fill="#193653")
    with Image.open(REPORT / "evaluation.png") as chart:
        chart.thumbnail((1100, 1000))
        chart_page.paste(chart.convert("RGB"), ((1240 - chart.width) // 2, 200))
    images.append(chart_page)
    images[0].save(REPORT / "report_draft.pdf", "PDF", resolution=150,
                   save_all=True, append_images=images[1:])
    (REPORT / "report_draft.md").write_text(markdown)
    print("Wrote report/report_draft.md and report/report_draft.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "video", "report"])
    parser.add_argument("--font", help="Japanese TTF/TTC/OTF font for video and report")
    args = parser.parse_args()
    if args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "video":
        render_video(args.font)
    else:
        make_report(args.font)


if __name__ == "__main__":
    main()
