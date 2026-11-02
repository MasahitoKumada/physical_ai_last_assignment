"""Frozen-observation comparisons and explanatory figures for the assignment.

The analytic baseline shares observations, physical feature normalization,
priors and curl shrinkage with the learned model. Only the fitted calibration
maps are replaced by the known equations. This isolates their contribution;
it is not a comparison against an estimator with no physical knowledge.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from collision_poc import (CASES, SEED, board_from, collect, compare,
                           perturb_board, source_hash, summarize)
from curved_poc import NOISE, window_from, with_noise
from game import GameConfig, GameSession, MODEL, ROOT
from physics import DT, ShotCommand
from predictor import IceEstimate, IceRegressor, ice_features, predict_board

REPORT = ROOT / "report"
METHODS = ("fixed", "analytic", "learned")


@dataclass(frozen=True)
class AnalyticIceModel:
    prior_friction: float
    prior_curl: float
    curl_prior_variance: float

    def estimate(self, window, previous=None):
        feature = ice_features(window)
        friction = float(np.clip(feature.deceleration / 9.81, .02, .09))
        prior = self.prior_curl if previous is None else previous.curl
        if not np.isfinite(prior) or not 0 <= prior <= .1:
            raise ValueError("Invalid previous curl estimate.")
        if feature.curvature is None:
            return IceEstimate(friction, prior)
        error = feature.curvature_standard_error
        variance = max(self.curl_prior_variance, 1e-12)
        weight = float(variance / (variance + error ** 2))
        curl = float(np.clip(prior + weight * (feature.curvature - prior), 0., .1))
        return IceEstimate(friction, curl, True, weight, float(error))


def evaluate_methods():
    model = IceRegressor.load(MODEL)
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    training = json.loads((REPORT / "game_curved_training.json").read_text())
    if digest != training["model_sha256"]:
        raise ValueError("Frozen model differs from its training record.")
    shots = collect()
    used = [r for split in ("train", "validation", "test")
            for r in json.loads((REPORT / f"game_curved_{split}_shots.json").read_text())["shots"]]
    for name in ("true_friction", "true_curl"):
        if not {s[name] for s in shots}.isdisjoint({s[name] for s in used}):
            raise ValueError("Comparison ice overlaps previous learning splits.")
    analytic = AnalyticIceModel(model.prior_friction, model.prior_curl, model.curl_prior_variance)
    prior = IceEstimate(model.prior_friction, model.prior_curl)
    rows, summaries = [], {}
    for noise_index, (noise, pos_std, vel_std, spin_std) in enumerate(NOISE):
        rng = np.random.default_rng(SEED + noise_index)
        group = []
        for shot in shots:
            window = with_noise(window_from(shot), pos_std, vel_std, spin_std, rng)
            board = perturb_board(board_from(shot["board"]), window, pos_std, vel_std, spin_std, rng)
            estimates = {"fixed": prior, "analytic": analytic.estimate(window), "learned": model.estimate(window)}
            # Freeze all forecasts before comparing with the recorded outcome.
            forecasts = {method: predict_board(board, estimate)[0] for method, estimate in estimates.items()}
            row = {"id": shot["id"], "case": shot["case"], "noise": noise,
                   "observed_time": board.time, "window_end_time": window[-1].time,
                   "estimates": {m: asdict(e) for m, e in estimates.items()},
                   **{m: compare(shot["result"], p) for m, p in forecasts.items()}}
            group.append(row)
        summaries[noise] = {"position_std_m": pos_std, "velocity_std_m_s": vel_std,
                            "spin_std_rad_s": spin_std,
                            "overall": {m: summarize(group, m) for m in METHODS},
                            "by_case": {c: {m: summarize([r for r in group if r["case"] == c], m)
                                            for m in METHODS} for c in CASES}}
        rows.extend(group)
        print(f"method comparison: {noise}, {len(group)} trials", flush=True)
    previous = json.loads((REPORT / "game_collision_evaluation.json").read_text())
    old = {(r["noise"], r["id"]): r for r in previous["rows"]}
    checks = {
        "96_trials_reused_in_three_noise_conditions": len(shots) == 96 and len(rows) == 288,
        "all_forecasts_use_same_pre_contact_observation": all(
            r["window_end_time"] <= r["observed_time"] + 1e-9 for r in rows)
            and all(not any(o["contact"] for o in s["observations"]) for s in shots),
        "fixed_and_learned_reproduce_existing_evaluation": all(
            r[m] == old[(r["noise"], r["id"])][m] for r in rows for m in ("fixed", "learned")),
        "finite_metrics": all(np.isfinite(s["overall"][m]["mean_stop_error_m"])
                              for s in summaries.values() for m in METHODS),
    }
    payload = {"status": "passed" if all(checks.values()) else "needs_review", "checks": checks,
               "model_sha256": digest, "physics_sha256": source_hash(), "seed": SEED,
               "independent_shots": len(shots), "noise_evaluations": len(rows),
               "prediction_time_after_release_seconds": .7,
               "analytic_definition": "mu = normalized deceleration / 9.81; kappa = normalized heading slope; same bounds, prior and residual shrinkage as learned model, without fitted calibration maps",
               "shared_prior": asdict(analytic),
               "protocol": "Prepared layouts, free/head-on/grazing/chain; one 0.6 s observation window; no future sweep; no online EMA in either estimator; same noisy board for all methods",
               "limitations": ["Same known physics in source and forecast; no real ice",
                               "Common priors and curl prior variance come from the frozen training model",
                               "96 existing trajectories reused at three synthetic noise levels, not 288 independent shots",
                               "In-flight collision benchmark, separate from the 24 complete-game evaluation",
                               "No model selection or retraining on these evaluation cases",
                               "Passing checks validates the comparison, not superiority of a method"],
               "summaries": summaries, "rows": rows}
    (REPORT / "game_method_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    if payload["status"] != "passed":
        raise RuntimeError("Comparison integrity check failed.")
    plot_methods(payload)
    return payload


def plot_methods(payload):
    from analysis import plotting
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    x = np.arange(3)
    for i, (method, label, color) in enumerate((("fixed", "Fixed prior", "#bc7452"),
                                              ("analytic", "Physics estimate", "#9878bc"),
                                              ("learned", "Learned estimate", "#23899d"))):
        for axis, key, title in zip(axes, ("mean_stop_error_m", "max_endpoint_error_m"),
                                     ("Mean error per in-rink stone", "Worst stone endpoint error")):
            values = [payload["summaries"][n]["overall"][method][key] * 100 for n, *_ in NOISE]
            bars = axis.bar(x + (i - 1) * .25, values, .25, label=label, color=color)
            axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
            axis.set_xticks(x, ["Exact", "Small noise", "Larger noise"])
            axis.set_ylabel("Error (cm)")
            axis.set_title(title)
            axis.margins(y=.25)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=1., facecolor="white", labelcolor="black")
    fig.suptitle("96 shared trajectories / frozen observations at 0.7 s / no future sweep")
    fig.savefig(REPORT / "game_method_comparison.png", dpi=180)
    plt.close(fig)


def make_demo_comparisons():
    record_path = REPORT / "game_demo_replay.json"
    record = json.loads(record_path.read_text())
    if record["model_sha256"] != hashlib.sha256(MODEL.read_bytes()).hexdigest():
        raise ValueError("Demo record uses another model.")
    model = IceRegressor.load(MODEL)
    prior = IceEstimate(model.prior_friction, model.prior_curl)
    rows = []
    for turn, shot in enumerate(record["history"], 1):
        # The first demo throw has future player sweeping and is not a fair
        # no-sweep pre-shot endpoint comparison. Use all remaining throws.
        if any(e["active"] for e in shot["sweep_events"]):
            continue
        board = board_from(shot["board_before"])
        command = ShotCommand(**shot["command"])
        estimate = IceEstimate(**shot["start_estimate"])
        forecasts = {m: predict_board(board, e, command)[0]
                     for m, e in (("fixed", prior), ("learned", estimate))}
        row = {"turn": turn, "observed_time": board.time, "command": asdict(command),
               "estimate_available_before_throw": asdict(estimate), "future_sweep": False,
               **{m: compare(shot["result"], p) for m, p in forecasts.items()}}
        rows.append(row)
        render_comparison(row).save(REPORT / f"game_comparison_turn_{turn}.png")
    mid = capture_demo_midflight(record, prior)
    render_comparison(mid).save(REPORT / "game_comparison_turn_2_inflight.png")
    payload = {"source": record_path.name, "source_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
               "selection": "All demo throws with no subsequent sweep; first throw excluded",
               "protocol": "Same board and command; estimates recorded before throw; actual result used only for scoring afterwards",
               "rows": rows, "second_throw_in_flight": mid}
    (REPORT / "game_demo_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def capture_demo_midflight(record, prior):
    """Replay executed controls; freeze second-throw predictions at 1.0 s."""
    session = GameSession(GameConfig(**record["config"]), adaptive=record["adaptive"])
    captured = None
    for turn, shot in enumerate(record["history"], 1):
        session.launch(ShotCommand(**shot["command"]))
        event_index, sweeping = 0, False
        events = shot["sweep_events"]
        while session.world.phase in ("DELIVERY", "SLIDING", "SETTLING"):
            while event_index < len(events) and events[event_index]["time"] - DT <= session.world.data.time + 1e-9:
                sweeping = events[event_index]["active"]
                event_index += 1
            session.step(sweeping)
            if (turn == 2 and captured is None and session.world.release is not None
                    and session.world.data.time - session.world.release >= 1.):
                board = session.world.observe_board()
                estimate = session.estimator.estimate
                if sweeping or any(e["active"] and e["time"] > board.time for e in events):
                    raise ValueError("No-future-sweep comparison requires matching controls.")
                forecasts = {m: predict_board(board, e)[0] for m,e in (("fixed", prior), ("learned", estimate))}
                captured = {"turn": 2, "timing": "in_flight", "after_release_seconds": board.time - session.world.release,
                            "observed_time": board.time, "estimate_available_at_prediction": asdict(estimate),
                            "last_observation_time": session.history[-1]["observations"][-1]["time"],
                            "future_sweep": False, "forecasts": forecasts}
        actual = session.world.result()
        if (actual["score"] != shot["result"]["score"] or
                compare(shot["result"], actual)["worst_stone_error_m"] > 1e-6):
            raise ValueError("Demo replay no longer matches its recorded result.")
        if turn == 2:
            if captured is None:
                raise ValueError("Missing comparison instant.")
            forecasts = captured.pop("forecasts")
            captured.update({m: compare(actual, p) for m,p in forecasts.items()})
            return captured



def render_comparison(row, failure=False):
    """A top-down explanatory plot of recorded points, not a GUI screenshot."""
    from PIL import Image, ImageDraw
    from player_view import fit
    image = Image.new("RGB", (1280, 720), "#10243b")
    draw = ImageDraw.Draw(image)
    in_flight = row.get("timing") == "in_flight"
    example = row.get("example_kind") == "head_on"
    title = "失敗例：かすり衝突で予測が外れる" if failure else ("事例：未知の氷で、正面衝突の行き先を予測" if example else ("2投目：新しい動きから、氷を読み直す" if in_flight else f"{row['turn']}投目：同じ狙いで、予測と結果を比較"))
    fit(draw, title, 45, 27, 1190, 38, "white")
    sub = "学習方式の最大誤差例 / 大きい合成観測ノイズ / 解放0.7秒後に予測" if failure else ("正確な観測 / 解放0.7秒後・衝突前に予測を固定 / その後のスイープなし" if example else ("解放1.0秒後の観測で予測を固定 / 同じ石の状態 / その後のスイープなし" if in_flight else "投げる前の推定を固定 / 同じ盤面・方向・パワー / その後のスイープなし"))
    fit(draw, sub, 45, 86, 1190, 23, "#b3cbdc")
    xmin = 0. if failure else 1.6
    for i, (method, label, color) in enumerate((("fixed", "標準の氷で予測", "#ffba79"),
                                               ("learned", "AIが読んだ氷で予測", "#51d9de"))):
        left = 35 + i * 625
        draw.rounded_rectangle((left, 139, left + 595, 595), radius=14, fill="#19334e")
        fit(draw, label, left + 20, 154, 550, 28, color)
        scale = min(550 / (4. - xmin), 310 / 1.5)
        def point(xy):
            return left + 22 + (xy[0] - xmin) * scale, 380 - xy[1] * scale
        cx, cy = point((3., 0.))
        for radius, fill in ((.45, "#2a6596"), (.30, "#244763"), (.15, "#6f3e58")):
            radius *= scale
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=fill)
        draw.line((cx-7, cy, cx+7, cy), fill="#c0d0df")
        draw.line((cx, cy-7, cx, cy+7), fill="#c0d0df")
        for stone in row[method]["stones"]:
            px, py = point(stone["predicted_position"])
            ax, ay = point(stone["actual_position"])
            r = .07 * scale
            draw.line((px, py, ax, ay), fill=color, width=2)
            draw.ellipse((px-r, py-r, px+r, py+r), outline=color, width=3)
            if stone["actual_alive"]:
                draw.ellipse((ax-r*.68, ay-r*.68, ax+r*.68, ay+r*.68), fill="white")
            else:
                draw.line((ax-6, ay-6, ax+6, ay+6), fill="white", width=3)
                draw.line((ax-6, ay+6, ax+6, ay-6), fill="white", width=3)
            fit(draw, str(stone['id'] + 1), ax + r + 3, ay - 10, 30, 18, "white")
        errors = [s["error_m"] for s in row[method]["stones"] if s["actual_alive"]]
        fit(draw, f"平均誤差 {np.mean(errors)*100:.2f} cm", left + 22, 530, 550, 26, "white")
        fit(draw, f"最大誤差 {row[method]['worst_stone_error_m']*100:.2f} cm", left + 22, 566, 550, 18, "#b3cbdc")
    fit(draw, "色つきの輪＝予想停止位置　白い点＝実際の停止位置　数字＝石の番号", 45, 618, 1190, 23, "white")
    footer = "事後に最大誤差例を選択。平均精度や任意の配置での成功を示す場面ではありません。" if failure else ("正面衝突テストの先頭の1例を表示。性能の評価は、別図の96投全体で確認します。" if example else ("新しい氷の観測を使って推定値を更新。学習モデルの重みはゲーム中に変えていません。" if in_flight else "前投の読みは、次の氷で外れることも。未知の変化を先に知ることはできません。"))
    fit(draw, footer, 45, 668, 1190, 20, "#b3cbdc")
    return image


def make_role_diagram():
    from PIL import Image, ImageDraw
    from player_view import fit
    image = Image.new("RGB", (1280, 720), "#10243b")
    draw = ImageDraw.Draw(image)
    fit(draw, "人が決める。ロボットが投げる。AIが氷を読む。", 45, 35, 1190, 38, "white")
    nodes = [(45, 158, "人の判断", "方向・パワー・回転を選ぶ", "スイープも人が操作", "#ffcf72"),
             (445, 158, "ロボットの制御", "モーターで保持・加速", "回転をつけて解放", "#9ac8ff"),
             (845, 158, "動きの観測", "石の位置・速度・回転", "接触中のデータは除外", "#9ac8ff"),
             (845, 383, "学習モデルの推論", "摩擦・カール係数を推定", "事前学習済みの重みは固定", "#51d9de"),
             (445, 383, "物理シミュレーション", "推定した氷で軌跡を計算", "衝突後の全石の位置を予測", "#c5a0ef"),
             (45, 383, "予測を見て、次の狙いへ", "停止位置・点数の目安", "投球の指示は自動変更しない", "#ffcf72")]
    for x,y,title,line1,line2,color in nodes:
        draw.rounded_rectangle((x,y,x+350,y+155),radius=16,fill="#19334e",outline=color,width=2)
        fit(draw,title,x+18,y+16,314,27,color)
        fit(draw,line1,x+18,y+67,314,22,"white")
        fit(draw,line2,x+18,y+108,314,20,"#b3cbdc")
    for x,y,glyph in [(405,210,"→"),(805,210,"→"),(1000,331,"↓"),(805,435,"←"),(405,435,"←"),(200,331,"↑")]:
        fit(draw,glyph,x,y,40,32,"white")
    fit(draw,"学習時：既知の氷でデータを作り、観測と係数の対応を学習。",45,580,1190,24,"white")
    fit(draw,"プレイ時：真の氷の値は渡さず、新しい観測から推定値を更新する。",45,625,1190,24,"white")
    fit(draw,"同じ簡略物理モデル内の検証。実氷・画像認識・実機への転用は未検証。",45,679,1190,20,"#b3cbdc")
    image.save(REPORT / "game_ai_roles.png")


def make_case_figures(result):
    failure = max((r for r in result["rows"] if r["noise"] == "larger_noise"),
                  key=lambda r: r["learned"]["worst_stone_error_m"])
    render_comparison(failure, failure=True).save(REPORT / "game_prediction_failure.png")
    example = next(r for r in result["rows"] if r["noise"] == "clean" and r["case"] == "head_on")
    render_comparison({**example, "example_kind": "head_on"}).save(REPORT / "game_prediction_example.png")
    (REPORT / "game_comparison_figure_sources.json").write_text(json.dumps({
        "source": "game_method_comparison.json",
        "source_sha256": hashlib.sha256((REPORT / "game_method_comparison.json").read_bytes()).hexdigest(),
        "example": {"id": example["id"], "noise": "clean", "selection": "First head-on case in the existing ordered test set, not selected by error"},
        "failure": {"id": failure["id"], "noise": "larger_noise", "selection": "Post-hoc maximum learned-method endpoint error across the 96 trials"},
    }, indent=2) + "\n")
    return failure


def build_evidence():
    result = evaluate_methods()
    demo = make_demo_comparisons()
    make_role_diagram()
    failure = make_case_figures(result)
    print(json.dumps({"status": result["status"], "mean_errors_cm": {
        n: {m: s["overall"][m]["mean_stop_error_m"] * 100 for m in METHODS}
        for n, s in result["summaries"].items()}, "demo_turns": [r['turn'] for r in demo['rows']],
        "failure": failure["id"]}, indent=2), flush=True)


if __name__ == "__main__":
    build_evidence()
