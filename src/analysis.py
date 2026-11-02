"""End-to-end evaluation and local submission artifacts for the curling game."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from game import DEMO_COMMANDS, GameConfig, GameSession, MODEL, ROOT
from predictor import IceEstimate, predict_board
from collision_poc import compare, summarize

REPORT = ROOT / "report"
VIDEO = ROOT / "video"


def plotting():
    cache = Path(tempfile.gettempdir()) / "curling-learning-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def evaluate_games():
    started = time.perf_counter()
    rows, games, latencies = [], [], []
    index = 0
    for initial in (.0413, .0468, .0542, .0591):
        for drift in (-.04, 0., .04):
            for curl in (.023, .037):
                session = GameSession(GameConfig(20261101 + index, initial, drift, curl))
                prior = IceEstimate(session.model.prior_friction, session.model.prior_curl)
                for turn, command in enumerate(DEMO_COMMANDS):
                    board = session.world.observe_board()
                    before_estimate = session.estimator.estimate
                    fixed_pre, _ = predict_board(board, prior, command)
                    before = time.perf_counter()
                    learned_pre, _ = predict_board(board, before_estimate, command)
                    latencies.append(time.perf_counter() - before)
                    session.launch(command)
                    middle = []
                    def capture(s):
                        if not middle and s.world.release is not None and s.world.data.time - s.world.release >= 1.:
                            middle.append((s.world.observe_board(), s.estimator.estimate))
                    actual = session.run_until_rest(callback=capture)
                    if not middle:
                        raise RuntimeError("Missing in-flight evaluation state.")
                    mid_board, mid_estimate = middle[0]
                    fixed_mid, _ = predict_board(mid_board, prior)
                    before = time.perf_counter()
                    learned_mid, _ = predict_board(mid_board, mid_estimate)
                    latencies.append(time.perf_counter() - before)
                    for timing, fixed, learned in (("pre_shot", fixed_pre, learned_pre), ("in_flight", fixed_mid, learned_mid)):
                        rows.append({"game": index, "turn": turn + 1, "timing": timing,
                                     "fixed": compare(actual, fixed), "learned": compare(actual, learned)})
                games.append({"config": asdict(session.config), "ice_sequence_evaluation_only": session._ice,
                              "history": session.history, "updates": session.estimator.events,
                              "final_score": session.world.result()["score"], "phase": session.world.phase})
                index += 1
                print(f"game evaluation: {index}/24", flush=True)
    summaries = {}
    for timing in ("pre_shot", "in_flight"):
        group = [r for r in rows if r["timing"] == timing]
        summaries[timing] = {"overall": {m: summarize(group, m) for m in ("fixed", "learned")},
                             "by_turn": {str(turn): {m: summarize([r for r in group if r["turn"] == turn], m)
                                                     for m in ("fixed", "learned")} for turn in (1, 2, 3)}}
    checks = {
        "24_complete_three_shot_games": len(games) == 24 and all(g["phase"] == "RESULT" and len(g["history"]) == 3 for g in games),
        "all_games_updated_from_observations": all(len(g["updates"]) > 0 for g in games),
        "first_pre_shot_uses_prior": all(r["fixed"] == r["learned"] for r in rows if r["turn"] == 1 and r["timing"] == "pre_shot"),
        "later_pre_shot_means_improve": all(summaries["pre_shot"]["by_turn"][str(t)]["learned"]["mean_stop_error_m"] < summaries["pre_shot"]["by_turn"][str(t)]["fixed"]["mean_stop_error_m"] for t in (2, 3)),
        "in_flight_mean_improves": summaries["in_flight"]["overall"]["learned"]["mean_stop_error_m"] < summaries["in_flight"]["overall"]["fixed"]["mean_stop_error_m"],
        "prediction_under_1s": max(latencies) < 1.,
    }
    payload = {"status": "passed" if all(checks.values()) else "needs_review", "checks": checks,
               "games": games, "rows": rows, "summaries": summaries,
               "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
               "independent_games": 24, "shots": 72, "predictions_per_method": 144,
               "max_prediction_seconds": max(latencies), "elapsed_seconds": time.perf_counter() - started,
               "observation": "Exact simulator position, velocity, spin and executed controls; no true ice passed to estimator",
               "in_flight_evaluation_time": "1.0 seconds after release",
               "limitations": ["Same simplified physics in game and forecast; no real-world validation",
                               "No synthetic noise in this integrated evaluation; separate collision noise tests are retained",
                               "Three scripted player commands, no sweeping in the paired prediction evaluation",
                               "Same commands yield same score; prediction aid does not automatically improve player score",
                               "Next shot retains last estimate and cannot know the unseen inter-shot ice change",
                               "Means do not guarantee accuracy on arbitrary layouts or grazing contacts"]}
    REPORT.mkdir(exist_ok=True)
    (REPORT / "game_evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, timing, title in zip(axes, ("pre_shot", "in_flight"), ("Before throwing", "1.0 s after release")):
        for method, label, color in (("fixed", "Fixed ice prior", "#bc7452"), ("learned", "Online correction", "#23899d")):
            axis.plot([1, 2, 3], [summaries[timing]["by_turn"][str(t)][method]["mean_stop_error_m"] * 100 for t in (1, 2, 3)],
                      "o-", label=label, color=color)
        axis.set_xticks([1, 2, 3])
        axis.set_xlabel("Throw")
        axis.set_ylabel("Mean error per in-rink stone (cm)")
        axis.set_title(title)
        axis.legend()
    fig.savefig(REPORT / "game_evaluation.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"status": payload["status"], "checks": checks,
                      "mean_errors_cm": {timing: {t: {m: v[m]["mean_stop_error_m"] * 100 for m in ("fixed", "learned")}
                                                  for t, v in s["by_turn"].items()} for timing, s in summaries.items()}}, indent=2), flush=True)
    if payload["status"] != "passed":
        raise RuntimeError("Integrated evaluation needs review; see game_evaluation.json.")
    return payload


def make_video():
    """Render an annotated replay of real game-engine states, at 1x motion."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from poc import PocRenderer, font_path
    from player_view import ice_hint, shot_outcome, fit
    from comparison import make_demo_comparisons
    evaluation = json.loads((REPORT / "game_evaluation.json").read_text())
    if evaluation["status"] != "passed":
        raise ValueError("Review the integrated evaluation before generating a demo.")
    comparison = json.loads((REPORT / "game_method_comparison.json").read_text())
    if comparison["status"] != "passed" or comparison["model_sha256"] != hashlib.sha256(MODEL.read_bytes()).hexdigest():
        raise ValueError("Run game.py compare with the current model first.")
    session = GameSession(GameConfig(20261101, .052, -.05, .03))
    renderer = PocRenderer(session.world, width=960, height=540)
    renderer.camera.distance = 3.5
    font = lambda size: ImageFont.truetype(font_path(), size)
    small, body, heading, title = (font(s) for s in (21, 27, 34, 46))
    VIDEO.mkdir(exist_ok=True)
    path = VIDEO / "game_demo.mp4"
    writer = imageio.get_writer(path, fps=25, codec="libx264", quality=8, macro_block_size=16,
                                ffmpeg_log_level="error", ffmpeg_params=["-pix_fmt", "yuv420p"])
    frames, chapters, predictions = 0, [], []

    def emit(frame, seconds=None):
        nonlocal frames
        count = 1 if seconds is None else round(seconds * 25)
        array = np.asarray(frame)
        for _ in range(count):
            writer.append_data(array)
        frames += count

    def card(name, lines, seconds, chart=None):
        chapters.append({"title": name, "start_seconds": frames / 25})
        frame = Image.new("RGB", (1280, 720), "#10243b")
        draw = ImageDraw.Draw(frame)
        fit(draw, name, 55, 48, 1170, 46, "white")
        if chart:
            with Image.open(chart) as figure:
                figure = figure.convert("RGB")
                figure.thumbnail((1120, 425))
                frame.paste(figure, ((1280 - figure.width) // 2, 140))
            y = 580
        else:
            y = 200
        for line in lines:
            fit(draw, line, 60, y, 1160, 27, "#d3e5f0")
            y += 55
        draw.text((60, 680), "Physical AI / MuJoCo / 直交3軸＋回転軸", font=small, fill="#88afc5")
        emit(frame, seconds)

    def diagram(name, path, seconds):
        chapters.append({"title": name, "start_seconds": frames / 25})
        with Image.open(path) as picture:
            emit(picture.convert("RGB"), seconds)

    def scene(turn, command, caption, frozen=False):
        frame = Image.new("RGB", (1280, 720), "#10243b")
        frame.paste(Image.fromarray(renderer.render()), (0, 90))
        draw = ImageDraw.Draw(frame)
        draw.text((25, 18), f"{turn} 投目 / 3　ロボット・カーリング", font=heading, fill="white")
        draw.text((975, 100), f"{turn} / 3 投", font=heading, fill="white")
        ice = ice_hint(session.estimator.estimate, session.model.prior_friction,
                       session.estimator.updates, session.completed, session.world.phase)
        forecast_score = f"予想合計  {renderer.prediction['score']} 点" if renderer.prediction else "3個の最終配置で採点"
        labels = [f"方向  {command.angle:+.0f}°", f"パワー  {command.power:.0f} / 100",
                  f"回転  { {-1: '右', 0: 'なし', 1: '左'}[command.spin] }",
                  f"スイープ  {'ON' if session.world.sweeping else 'OFF'}",
                  f"現在の合計  {session.world.result()['score']} 点", forecast_score, "",
                  ice[0] if session.estimator.updates else "AI：まずは氷を観測",
                  "半透明：止まる場所の目安", "点線：予想軌跡"]
        for i, line in enumerate(labels):
            fit(draw, line, 975, 160 + 36 * i, 290, 21, "#d3e5f0")
        draw.text((25, 646), caption, font=body, fill="white")
        draw.text((25, 687), "操作記録の再生 / 説明中は静止" if frozen else "操作記録の再生 / 動作部分：等速 / 予測は『ここからスイープなし』",
                  font=small, fill="#a5c7d9")
        return frame

    try:
        card("狙って、滑らせて、中心へ。", ["3投の最終配置でスコアを競う、1人用ロボット・カーリング。",
             "止まる場所を見ながら、方向とパワーを少しずつ調整。",
             "思い描いた場所に寄せて、同じ氷でベスト更新を狙おう。"], 4)
        card("置いたストーンも、次の1投に使う", ["1ゲーム3投。最後に中心へ集まった石ほど高得点。",
             "ハウスの外周にかかれば1点。場外の石は除外する。",
             "自分の石を押し込み、3個の最終配置を考える。"], 6)
        card("AIが氷を読む。次の狙いの手がかりに。", ["半透明の石は、スイープしない場合の予想停止位置。",
             "AIが滑り方から氷を推定し、物理計算で行き先を読む。",
             "まだ観測していない氷の変化は分からない。予測は目安。"], 6)
        diagram("AI・物理計算・ロボットの役割", REPORT / "game_ai_roles.png", 8)
        for turn, command in enumerate(DEMO_COMMANDS, start=1):
            before_turn = session.world.result()
            chapters.append({"title": f"{turn}投目", "start_seconds": frames / 25})
            renderer.prediction, renderer.predicted_trace = session.forecast(command, True)
            captions = {1: "1投目：中央を狙う。途中で短くスイープする。",
                        2: "2投目：前の石に当て、中心へ押し込む。",
                        3: "3投目：残った配置を使い、合計点を伸ばす。"}
            emit(scene(turn, command, captions[turn], True), 4)
            session.launch(command)
            last_prediction = -float("inf")
            step = 0
            while session.world.phase in ("DELIVERY", "SLIDING", "SETTLING"):
                elapsed = -1. if session.world.release is None else session.world.data.time - session.world.release
                session.step(turn == 1 and .25 <= elapsed < .45)
                step += 1
                if session.world.phase == "SLIDING" and session.world.data.time - last_prediction >= .4:
                    renderer.prediction, renderer.predicted_trace = session.forecast(include_trace=True)
                    last_prediction = float(session.world.data.time)
                    predictions.append({"time": last_prediction, "estimate": asdict(session.estimator.estimate),
                                        "prediction": renderer.prediction})
                if step % 20 == 0:
                    if session.world.phase == "DELIVERY":
                        caption = "ロボットのモーターで保持・加速・回転・解放。"
                    elif session.world.sweeping:
                        caption = "スイープ中：抵抗を小さくし、もう少し先へ。"
                    else:
                        caption = "動きを観測して予測を更新。衝突した石の行き先も計算。"
                    frame = scene(turn, command, caption)
                    emit(frame)
                    if turn == 3 and session.world.release is not None and 1.0 <= elapsed < 1.04:
                        frame.save(REPORT / "game_video_preview.png")
                        Image.fromarray(renderer.render()).save(REPORT / "game_scene_preview.png")
            renderer.prediction, renderer.predicted_trace = None, []
            outcome = shot_outcome(before_turn, session.world.result(), session.world.current)
            emit(scene(turn, command, f"{outcome['title']}  {outcome['delta']:+d}点 / {outcome['detail']}", True), 3)
            if turn == 1:
                card("投球ごとに、氷の滑りやすさが変わる", ["このデモでは、次の投球の氷が少しずつ滑りやすくなる。",
                     "AIは前の推定を引き継ぎ、新しい動きで補正する。",
                     "未来の氷の変化を、正解として教えることはしない。"], 7)
        score = session.world.result()["score"]
        card(f"3投終了：合計 {score} 点", ["最終配置で採点。後の衝突で動いた石も採点し直す。",
             "次はベスト更新へ。同じ氷で、狙いを少し変えて再挑戦。"], 6)
        session.save_record(REPORT / "game_demo_replay.json")
        make_demo_comparisons()
        diagram("2投目：投球前の予測と結果", REPORT / "game_comparison_turn_2.png", 10)
        diagram("2投目：解放1.0秒後の観測で補正", REPORT / "game_comparison_turn_2_inflight.png", 10)
        diagram("事例：未学習の氷で正面衝突を予測", REPORT / "game_prediction_example.png", 9)
        pre = evaluation["summaries"]["pre_shot"]["by_turn"]
        line = f"投球前の平均誤差：2投目 {pre['2']['fixed']['mean_stop_error_m']*100:.1f}→{pre['2']['learned']['mean_stop_error_m']*100:.1f} cm、3投目 {pre['3']['fixed']['mean_stop_error_m']*100:.1f}→{pre['3']['learned']['mean_stop_error_m']*100:.1f} cm"
        card("24ゲーム・72投で予測を比較", [line, "入力は同一。得点向上ではなく、予測誤差の改善を評価。"], 10, REPORT / "game_evaluation.png")
        values = comparison["summaries"]["small_noise"]["overall"]
        baseline_line = "小さいノイズの平均誤差：" + "、".join(
            f"{label} {values[m]['mean_stop_error_m']*100:.2f} cm"
            for m, label in (("fixed", "固定"), ("analytic", "物理式"), ("learned", "学習")))
        card("物理式とも比較：96投・3種類の観測条件", [baseline_line,
             "この既知の物理法則では両者はほぼ同等。学習の優位性は確認できない。"], 14, REPORT / "game_method_comparison.png")
        diagram("失敗例：大きいノイズのかすり衝突", REPORT / "game_prediction_failure.png", 9)
        card("できたことと、残る限界", ["3投・衝突・回転・スイープと、氷の推定・逐次補正を統合。",
             "大きい観測ノイズのかすり衝突では、最大約66 cm外れる例。",
             "簡略化した同一モデル内の検証。実機・実氷への転用は未検証。"], 8)
        card("人が狙い、ロボットが投げ、AIが氷を読む。", ["ロボット・カーリング", "MuJoCo / 教師あり回帰 / 観測からの逐次推定"], 5)
    finally:
        writer.close()
        renderer.close()
    session.save_record(REPORT / "game_demo_replay.json")
    (REPORT / "game_demo_predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    metadata = {"path": "video/game_demo.mp4", "duration_seconds": frames / 25, "fps": 25,
                "frames": frames, "width": 1280, "height": 720, "motion_playback_speed": 1,
                "description": "Annotated scripted input replay through the same GameSession used by the UI; cards and aim pauses are static",
                "learned_ai": True, "online_correction": True, "demo_config": asdict(session.config),
                "final_score": session.world.result()["score"], "chapters": chapters,
                "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(), "submitted": False}
    (REPORT / "game_video_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if not 60 <= metadata["duration_seconds"] <= 180:
        raise RuntimeError("Video duration is outside the assignment range.")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return metadata


def make_report(account=None):
    from PIL import Image, ImageDraw, ImageFont
    from poc import font_path
    evaluation = json.loads((REPORT / "game_evaluation.json").read_text())
    video = json.loads((REPORT / "game_video_metadata.json").read_text())
    comparison = json.loads((REPORT / "game_method_comparison.json").read_text())
    if comparison["status"] != "passed":
        raise ValueError("Review the method comparison before generating the report.")
    pre = evaluation["summaries"]["pre_shot"]["by_turn"]
    mid = evaluation["summaries"]["in_flight"]["by_turn"]
    result_text = "、".join(f"{t}投目 {pre[str(t)]['fixed']['mean_stop_error_m']*100:.2f}→{pre[str(t)]['learned']['mean_stop_error_m']*100:.2f} cm" for t in (1, 2, 3))
    pages = [
        ("ロボット・カーリング", [
            "1. 概要と目的",
            "投球ごとに滑りやすさが変わる氷の上で、直交ロボットが3個のストーンを投げる1人用ゲーム。プレイヤーが方向・パワー・回転とスイープを操作し、AIが軌跡と衝突後の全ストーンの停止位置を予測して判断を支援する。",
            "3投終了後の最終配置で採点する。中心は100点、距離に応じて直線的に減点し、ハウス外周にかかる石は1点、完全に外なら0点。既存の石を押し込む戦略を楽しめる独自ルールで、公式競技の採点そのものではない。",
            "2. ロボットと物理の実装",
            "Python 3.12とMuJoCo 3.17.0を使用。直交3軸と鉛直回転軸をモーターで制御し、weld拘束で保持した石を加速・回転して解放する。本番の投球結果を作るために石へ初速を直接設定していない。",
            "石は水平2軸と回転1軸で動く平面モデル。円形同士の接触はMuJoCoで扱い、摩擦・カール・スイープは簡略化した力として計算する。氷の融解や微細な表面構造の再現は対象外。",
            "プレイ画面はハウスの拡大、方向0.5度・パワー1ずつの微調整、1投後の予想合計点に対応する。停止後は中心までの距離・増減点・各石の点数を表示し、同じ氷で再挑戦できる。新しい表示による楽しさの向上は未評価。",
        ], "game_player_preview.png" if (REPORT / "game_player_preview.png").exists() else "game_scene_preview.png"),
        ("AIの役割と評価", [
            "3. 事前学習とゲーム中補正",
            "学習189投、調整72投で、自由滑走の減速率と方向変化から摩擦・カール係数を推定する小規模な教師あり回帰を学習した。既知のスイープ作用と回転・速度の関係を利用する。モデル重みはゲーム中に変えず、0.6秒の観測窓から推定値を逐次補正する。",
            "真の氷パラメータは推定器へ渡さない。保持中・接触を含む区間・停止直前を除外し、回転がなくカールを識別できない場合は前の推定を保持する。独立した予測世界に観測可能な状態だけを復元し、『これ以降スイープなし』で全石を先読みする。",
            "役割の分担：人が投球を決める→ロボットが保持・加速・解放する→動きを観測する→学習モデルが氷を推定する→物理計算が停止位置を予測する→人が次の狙いを調整する。係数の推定と軌跡の計算を分けた構成である。",
            "4. 3投ゲームでの比較",
            "氷の初期値・増減・カール係数を変えた24ゲーム（72投）で、固定した氷を仮定する予測と、補正する予測を同一の入力・盤面で比較した。位置・速度はシミュレータから直接取得し、この統合評価では観測ノイズを加えていない。",
            f"投球前の平均停止位置誤差（固定→補正）：{result_text}。場内で停止した各石をIDで対応付けて計測した。1投目はまだ観測がないため両者が同じになる。",
        ], "game_evaluation.png"),
        ("工夫・限界・再現方法", [
            "5. 工夫した点",
            "氷の変化とスイープの影響を分離し、次投へ推定を引き継ぐ。リトライでは同じ氷の系列と盤面に戻し、オンライン補正だけを初期化する。AIは人の投球コマンドを変更しない。描画と先読みを分け、古い入力に対する予測を画面へ反映しない。",
            "初期の3D円柱接触では微小な摩擦変更でも連鎖衝突の停止位置が大きく変わるケースがあったため、比較実験を行い平面モデルへ変更した。見た目の3Dとロボットの保持・解放は維持し、数値的な不安定さを減らした。",
            "6. 失敗例と限界",
            "別の衝突検証96投では、小さい合成観測ノイズの平均誤差が正面2.71 cm、かすり2.58 cm、連鎖1.76 cm。大きいノイズではかすり衝突に最大65.84 cmの誤差と接触の見落としが残った。平均だけで予測の確実性を判断しない。",
            "本番と予測は同じ物理法則を共有し、画像認識・実機・実氷での検証は行っていない。同じ入力なら得点も同じなので、この固定入力の数値評価は予測精度の比較であり、人の得点向上を示すものではない。多関節アーム・スイープ助言は追加していない。",
            "7. 再現と成果物",
            "READMEに環境構築と操作を記載。game.pyから起動・評価・動画・説明資料の再生成を実行できる。学習モデルと評価記録はreport/に保存している。コード作成・検証・資料整備に生成AIの支援を使用した。",
            f"動画：video/game_demo.mp4（{video['duration_seconds']:.2f}秒、1280×720、25 fps）。同じゲーム実装での入力記録再生で、動作部分は等速。提出先への登録や公開は行っていない。",
        ], None),
    ]
    playtest_path = REPORT / "game_playtest_evaluation.json"
    if playtest_path.exists():
        study = json.loads(playtest_path.read_text())
        if study["source"] != "human_play" or not study["completed"] or not study["all_six_replays_matched"]:
            raise ValueError("Only completed, verified human playtests can enter the report.")
        pair_text = "、".join(f"{pair['pair']}組 {pair['score_without']}→{pair['score_with']}点（{pair['score_difference']:+d}点）"
                              for pair in study["pairs"])
        rating_text = "、".join(
            f"{label} {np.mean([g['ratings'][key] for g in study['games'] if not g['forecast']]):.1f}→"
            f"{np.mean([g['ratings'][key] for g in study['games'] if g['forecast']]):.1f}"
            for key, label in (("enjoyment", "面白さ"), ("agency", "自分で考えた感覚"), ("clarity", "操作の分かりやすさ")))
        plt = plotting()
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
        x = np.arange(len(study["pairs"]))
        for axis, field, title, ymax in ((axes[0], "score", "Final score", 300),
                                         (axes[1], "enjoyment", "Enjoyment rating (1-5)", 5)):
            for offset, suffix, label, color in ((-.18, "without", "No forecast", "#bc7452"),
                                                 (.18, "with", "Forecast shown", "#23899d")):
                bars = axis.bar(x + offset, [pair[f"{field}_{suffix}"] for pair in study["pairs"]], .34, label=label, color=color)
                axis.bar_label(bars, padding=2)
            axis.set_xticks(x, [f"Ice pair {pair['pair']}" for pair in study["pairs"]])
            axis.set_ylim(0, ymax * 1.1)
            axis.set_title(title)
        axes[0].legend()
        fig.suptitle("Exploratory human playtest: one participant, four games")
        fig.savefig(REPORT / "game_playtest_evaluation.png", dpi=180)
        plt.close(fig)
        pages.append(("人が遊んだときの比較", [
            "8. 比較方法",
            "本人1名が練習2ゲームの後、本番4ゲーム（各3投）を実施した。同じ氷の2組で予測表示あり・なしを比較し、A組はあり→なし、B組はなし→ありの順とした。投球操作は本人が自由に選択し、各ゲーム後に5段階の感想を回答した。練習は集計から除外した。",
            "9. 結果と解釈",
            f"最終得点（なし→あり）：{pair_text}。2組の平均差は{study['mean_score_difference']:+.1f}点だった。今回の4ゲームでは予測あり条件の得点が高かったが、人数・試行数が少なく、慣れや氷を経験した記憶、期待の影響が残る。一般的な効果や因果関係は断定しない。",
            f"5段階の感想の平均（なし→あり）：{rating_text}。本番では全項目が3／5で、楽しさや自分で考える感覚の向上は確認できなかった。得点と主観的な楽しさを分けて評価した。",
            "10. 実プレイで見つかった改善点",
            "投球中に画面全体が揺れるとの指摘があった。予測メッセージの切替で映像が上下8ピクセル動くことを再現し、配置とメッセージ領域を固定した。36通りの表示切替で移動がないことを確認した。改修後、本人から操作性が改善したように思うとの感想を得たが、5段階評価の再測定はしていない。",
            "中断なしで全ゲームが完了し、操作記録から停止位置・得点を再現できた。本番は回転なし・スイープなし。この比較には予測ガイドの有無の効果も含まれ、学習モデルだけの寄与は分離していない。改修後の感想は別の定性的な確認であり、上記の点数・評価値を更新する根拠にはしない。",
        ], "game_playtest_evaluation.png"))
    method_lines = []
    for noise, label in (("clean", "正確な観測"), ("small_noise", "小さい合成ノイズ"), ("larger_noise", "大きい合成ノイズ")):
        values = comparison["summaries"][noise]["overall"]
        method_lines.append(label + "：" + "、".join(
            f"{name} {values[m]['mean_stop_error_m']*100:.3f} cm"
            for m, name in (("fixed", "固定"), ("analytic", "物理式"), ("learned", "学習"))) + "。")
    pages.append(("物理式と学習の寄与を比較", [
        "11. 学習を使う意味の検証",
        "固定した氷との比較だけでは、観測に応じて補正する効果と学習する効果を分けられない。物理式から摩擦・カール係数を求める対照を追加した。摩擦はスイープの作用を除いた減速率を9.81で割り、カール係数は回転・速度で正規化した進行方向の変化から求める。",
        "3方式には同じ観測と盤面を渡す。物理式と学習モデルで特徴量、係数の範囲、事前値、カール推定の平滑化を共通にし、学習した変換係数の有無を比較する。共通の事前値と分散は既存の学習モデル由来であり、事前情報がない推定器との比較ではない。",
        "12. 条件と結果",
        "未学習の氷8組、4配置、3回転の既存96投を使用した。解放0.7秒後までの観測を固定し、その後はスイープなし。3段階の観測ノイズで同じ96投を再利用したため、288投の独立実験ではない。既存の固定・学習方式の結果との一致も検証した。",
        "場内で停止した石の平均誤差：" + " ".join(method_lines),
        "既知の物理法則と共通の特徴量を使うこの条件では、物理式と学習の誤差は近かった。観測による補正の有効性は示せるが、学習が不可欠・物理式より優位とは主張しない。未知の物理、実センサーや実氷への頑健性は未検証である。",
    ], "game_method_comparison.png"))
    images, markdown = [], "# ロボット・カーリング\n\n更新日：2026年11月2日\n\n"
    heading_font = ImageFont.truetype(font_path(), 30)
    body_font = ImageFont.truetype(font_path(), 25)
    title_font = ImageFont.truetype(font_path(), 44)
    for number, (name, paragraphs, figure) in enumerate(pages, start=1):
        image = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(image)
        draw.text((75, 65), name, font=title_font, fill="#163751")
        y = 155
        for paragraph in paragraphs:
            is_heading = len(paragraph) < 35 and paragraph[:1].isdigit()
            font = heading_font if is_heading else body_font
            line = ""
            for char in paragraph:
                if draw.textlength(line + char, font=font) > 1090:
                    draw.text((75, y), line, font=font, fill="#243f50")
                    y += 39
                    line = ""
                line += char
            draw.text((75, y), line, font=font, fill="#243f50")
            y += 58 if is_heading else 62
            markdown += ("## " if is_heading else "") + paragraph + "\n\n"
        if figure:
            with Image.open(REPORT / figure) as chart:
                chart = chart.convert("RGB")
                chart.thumbnail((1090, 1640 - y))
                image.paste(chart, ((1240 - chart.width) // 2, y + 5))
                y += chart.height + 5
            markdown += f"![図]({figure})\n\n"
        if y > 1655:
            raise RuntimeError(f"Report page {number} overflow ({y}).")
        draw.text((1110, 1680), str(number), font=body_font, fill="#658295")
        image.save(REPORT / f"game_report_page_{number}.png")
        images.append(image)
    generic = REPORT / "game_report_draft.pdf"
    images[0].save(generic, "PDF", resolution=150, save_all=True, append_images=images[1:],
                   title="ロボット・カーリング", author="", creator="Robot Curling")
    if account is not None and (not account.strip() or any(c in account for c in '/\\\n\r\x00')):
        raise ValueError("Account must be a filename-safe OmniCampus name or ID.")
    label = account or "アカウント名未設定"
    submission = REPORT / f"説明資料_PAI最終課題_{label}.pdf"
    submission.write_bytes(generic.read_bytes())
    (REPORT / "game_report_draft.md").write_text(markdown)
    (REPORT / "game_submission_metadata.json").write_text(json.dumps({
        "account": account, "report": submission.name, "youtube_title": f"PAI最終課題_{label}",
        "submission_url": "https://edu.omnicamp.us/courses/172/reports/130/", "submitted": False,
        "github_published": False, "youtube_uploaded": False, "account_name_pending": account is None,
        "report_pages": len(images),
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {generic.name} and {submission.name}", flush=True)
