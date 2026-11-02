"""Small, local human playtest: two practice games and four paired games.

Automated GUI checks are explicitly labelled and cannot supply human evidence.
Only the forecast display changes between conditions; controls and physics do not.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np

from game import GameConfig, GameSession, MODEL, ROOT

QUESTIONS = {
    "enjoyment": "このゲームは面白かった",
    "agency": "自分で狙いや戦略を考えて操作できた",
    "clarity": "操作方法が分かりやすかった",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint():
    names = ["game.py", "physics.py", "predictor.py", "ui.py", "player_view.py", "playtest.py", "game_scene.xml"]
    return {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() for name in names}


def plan_trials(seed):
    rng = np.random.default_rng(seed)
    first = bool(rng.integers(0, 2))
    practice = asdict(GameConfig(seed, .05, -.03, .03))
    # Fixed before playing; do not select environments based on score outcomes.
    configs = [asdict(GameConfig(seed + 1, float(rng.uniform(.043, .047)), -.04, .027)),
               asdict(GameConfig(seed + 2, float(rng.uniform(.054, .058)), .04, .035))]
    trials = [{"practice": True, "pair": "practice", "forecast": condition, "config": practice}
              for condition in (False, True)]
    for pair, config, order in zip(("A", "B"), configs, ((first, not first), (not first, first))):
        trials.extend({"practice": False, "pair": pair, "forecast": condition, "config": config}
                      for condition in order)
    return [{"index": i, **trial} for i, trial in enumerate(trials)]


class PlaytestStudy:
    def __init__(self, directory=None, seed=20261101, automated=False):
        self.automated = automated
        if directory is None:
            base = ROOT / "report" / ("playtest_smoke" if automated else "playtests")
            label = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            directory = base / label
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "study.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data["source"] != ("automated_gui_check" if automated else "human_play"):
                raise ValueError("Human and automated sessions must use different directories.")
            if self.data["code_sha256"] != fingerprint() or self.data["model_sha256"] != hashlib.sha256(MODEL.read_bytes()).hexdigest():
                raise ValueError("Code or model changed during this study. Start a new study to avoid mixing versions.")
        else:
            self.data = {
                "schema_version": 1, "created_at": now(), "source": "automated_gui_check" if automated else "human_play",
                "participant_count": 0 if automated else 1, "seed": seed, "trials": plan_trials(seed),
                "code_sha256": fingerprint(), "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                "attempts": [], "completed": [], "pending": None, "active": None,
                "rating_scale": "1=全くそう思わない, 2=あまりそう思わない, 3=どちらともいえない, 4=ややそう思う, 5=とてもそう思う",
                "limitations": ["One-person exploratory comparison; not population evidence or a causal estimate",
                                "Two ice pairs, opposite display order; learning and carryover can remain",
                                "Forecast presence is visible; expectation bias is possible",
                                "Practice excluded; no best-run selection; interruptions retained",
                                "Times include reading and thinking; pauses are not removed"],
            }
            self.save()

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.path)

    @property
    def next_trial(self):
        completed = {row["trial_index"] for row in self.data["completed"]}
        return next((trial for trial in self.data["trials"] if trial["index"] not in completed), None)

    def start_attempt(self):
        if self.data["pending"] is not None:
            raise ValueError("Rate the finished game before starting another one.")
        if self.next_trial is None:
            raise ValueError("All trials are already complete.")
        if self.data["active"] is not None:
            self.interrupt("Application ended before the previous attempt was completed.")
        number = len(self.data["attempts"]) + 1
        active = {"attempt": number, "trial_index": self.next_trial["index"], "started_at": now(),
                  "record": f"trial_{self.next_trial['index'] + 1:02d}_attempt_{number:02d}.json", "status": "active"}
        self.data["attempts"].append(active)
        self.data["active"] = number
        self.save()
        return self.directory / active["record"]

    def interrupt(self, reason):
        if self.data["active"] is not None:
            row = self.data["attempts"][self.data["active"] - 1]
            row.update(status="interrupted", reason=reason, ended_at=now())
            self.data["active"] = None
            self.save()

    def capture(self, session, elapsed, decision_seconds, forecasts, errors):
        if session.world.phase != "RESULT" or session.completed != 3 or len(decision_seconds) != 3:
            raise ValueError("Only a complete, three-shot game can be rated.")
        if self.data["active"] is None or self.data["pending"] is not None:
            raise ValueError("No active game, or this result was already recorded.")
        attempt = self.data["attempts"][self.data["active"] - 1]
        trial = self.data["trials"][attempt["trial_index"]]
        if asdict(session.config) != trial["config"]:
            raise ValueError("Game environment differs from the predefined trial.")
        session.save_record(self.directory / attempt["record"])
        row = {"trial_index": trial["index"], "attempt": attempt["attempt"], "practice": trial["practice"],
               "pair": trial["pair"], "forecast": trial["forecast"], "record": attempt["record"],
               "score": session.world.result()["score"], "elapsed_seconds": float(elapsed),
               "decision_seconds": list(decision_seconds), "forecast_completions": int(forecasts),
               "ui_errors": list(errors), "finished_at": now()}
        attempt.update(status="awaiting_ratings", ended_at=now())
        self.data["pending"], self.data["active"] = row, None
        self.save()

    def rate(self, ratings, comment):
        if set(ratings) != set(QUESTIONS) or any(type(value) is not int or not 1 <= value <= 5 for value in ratings.values()):
            raise ValueError("Choose a rating from 1 to 5 for every question.")
        if self.data["pending"] is None:
            raise ValueError("There is no completed game waiting for ratings.")
        row = self.data["pending"]
        row.update(ratings=ratings, comment=str(comment), rated_at=now())
        self.data["completed"].append(row)
        self.data["attempts"][row["attempt"] - 1]["status"] = "completed"
        self.data["pending"] = None
        self.save()


def summarize_study(directory):
    directory = Path(directory)
    data = json.loads((directory / "study.json").read_text())
    human = data["source"] == "human_play"
    main = [row for row in data["completed"] if not row["practice"]]
    pairs = []
    for pair in ("A", "B"):
        group = [row for row in main if row["pair"] == pair]
        if len(group) != 2 or {row["forecast"] for row in group} != {False, True}:
            continue
        on, off = next(row for row in group if row["forecast"]), next(row for row in group if not row["forecast"])
        pairs.append({"pair": pair, "order": ["あり" if row["forecast"] else "なし" for row in group],
                      "score_without": off["score"], "score_with": on["score"], "score_difference": on["score"] - off["score"],
                      "enjoyment_without": off["ratings"]["enjoyment"], "enjoyment_with": on["ratings"]["enjoyment"],
                      "rating_differences": {key: on["ratings"][key] - off["ratings"][key] for key in QUESTIONS},
                      "decision_seconds_difference": sum(on["decision_seconds"]) - sum(off["decision_seconds"])})
    complete = len(main) == 4 and len(pairs) == 2
    interrupted = sum(row["status"] == "interrupted" for row in data["attempts"])
    errors = [error for row in main for error in row["ui_errors"]]
    display_failures = [row["trial_index"] for row in main
                        if (row["forecast"] and row["forecast_completions"] == 0)
                        or (not row["forecast"] and row["forecast_completions"] != 0)]
    means = None
    if complete and human and not errors and not display_failures:
        means = {"score_difference": float(np.mean([pair["score_difference"] for pair in pairs])),
                 "rating_differences": {key: float(np.mean([pair["rating_differences"][key] for pair in pairs])) for key in QUESTIONS}}
    result = {"source": data["source"], "human_evidence": human and complete and not errors and not display_failures,
              "complete": complete, "main_games_completed": len(main), "pairs": pairs,
              "mean_with_minus_without": means, "interrupted_attempts": interrupted,
              "ui_errors": errors, "display_failures": display_failures, "limitations": data["limitations"]}
    lines = ["# AI予測あり・なしの比較プレイ", "",
             "本人1名の探索的な比較です。一般的な効果や因果関係を示すものではありません。" if human else
             "自動動作確認です。感想はダミー値で、人の得点改善・面白さの根拠には使えません。", "",
             f"本番の完了：{len(main)} / 4ゲーム。練習は集計から除外。中断した試行：{interrupted}件。", "",
             "| 組 | 順番 | 得点 なし / あり | 差（あり−なし） | 面白さ なし / あり |",
             "| --- | --- | --- | --- | --- |"]
    for pair in pairs:
        lines.append(f"| {pair['pair']} | {'→'.join(pair['order'])} | {pair['score_without']} / {pair['score_with']} | {pair['score_difference']:+d} | {pair['enjoyment_without']} / {pair['enjoyment_with']} |")
    if means is not None:
        lines.extend(["", f"2組の平均得点差（あり−なし）：{means['score_difference']:+.1f}点。",
                      f"面白さの回答差の平均：{means['rating_differences']['enjoyment']:+.1f}（5段階）。",
                      "順番を逆にしても、同じ氷を経験した記憶・慣れ・予測への期待の影響は残ります。",
                      "差が出ない、低下する、組ごとに逆になる結果もそのまま記録します。"])
    if not complete:
        lines.extend(["", "未完了です。欠けたゲームや感想を補って効果を判断しません。"])
    if errors or display_failures:
        lines.extend(["", "操作画面または予測表示に問題が記録されています。比較結果の解釈前に確認が必要です。"])
    lines.extend(["", "## 各ゲームの記録", "",
                  "| 本番 | 予測 | 得点 | 面白さ | 自分で考えた | 操作の分かりやすさ | 選択時間合計 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in main:
        rating = row["ratings"]
        lines.append(f"| {row['trial_index'] - 1} | {'あり' if row['forecast'] else 'なし'} | {row['score']} | {rating['enjoyment']} | {rating['agency']} | {rating['clarity']} | {sum(row['decision_seconds']):.1f}秒 |")
    lines.extend(["", "選択時間は投球可能になってから投球するまで。説明を読む時間や離席も含み、感想入力の時間は除外します。", ""])
    for row in main:
        if row["comment"].strip():
            lines.extend([f"本番{row['trial_index'] - 1}の自由記述：", "", row["comment"].strip(), ""])
    result["text"] = "\n".join(lines) + "\n"
    (directory / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (directory / "summary.md").write_text(result["text"])
    return result


def feedback(root, study):
    import tkinter as tk
    from tkinter import ttk
    dialog = tk.Toplevel(root)
    dialog.title("今回の3投の感想")
    dialog.geometry("650x510")
    if root.winfo_viewable():
        dialog.transient(root)
    panel = ttk.Frame(dialog, padding=20)
    panel.pack(fill="both", expand=True)
    ttk.Label(panel, text="今回の3投について教えてください", font=("Hiragino Sans", 17)).pack(anchor="w")
    ttk.Label(panel, text="1：全くそう思わない　3：どちらともいえない　5：とてもそう思う").pack(anchor="w", pady=10)
    variables = {}
    for key, question in QUESTIONS.items():
        ttk.Label(panel, text=question).pack(anchor="w", pady=(7, 2))
        line = ttk.Frame(panel)
        line.pack(anchor="w")
        value = variables[key] = tk.IntVar(value=0)
        for number in range(1, 6):
            ttk.Radiobutton(line, text=str(number), value=number, variable=value).pack(side="left", padx=13)
    ttk.Label(panel, text="良かった点・困った点（任意）").pack(anchor="w", pady=(12, 3))
    comment = tk.Text(panel, height=3, width=65)
    comment.pack(fill="x")
    warning = tk.StringVar()
    ttk.Label(panel, textvariable=warning, foreground="#a33b24").pack(anchor="w", pady=5)
    accepted = []
    def submit():
        try:
            study.rate({key: variable.get() for key, variable in variables.items()}, comment.get("1.0", "end-1c"))
        except ValueError:
            warning.set("3つの質問すべてで、1〜5を選んでください。")
            return
        accepted.append(True)
        dialog.destroy()
    ttk.Button(panel, text="保存して次へ", command=submit).pack(anchor="e")
    dialog.grab_set()
    if study.automated:
        def fill_dummy():
            for value in variables.values():
                value.set(3)
            comment.insert("1.0", "自動GUI確認用のダミー回答。人の評価ではありません。")
            submit()
        dialog.after(150, fill_dummy)
    root.wait_window(dialog)
    return bool(accepted)


def make_window(study, record_path):
    from poc import PocRenderer
    from ui import PocWindow

    class StudyWindow(PocWindow):
        def __init__(self):
            self.trial = study.next_trial
            self.captured = self.feedback_open = self.advance = False
            self.decision_seconds = []
            super().__init__(PocRenderer, session=GameSession(GameConfig(**self.trial["config"])),
                             autoplay=study.automated, prediction_enabled=self.trial["forecast"], record_path=record_path)
            number = self.trial["index"] + 1 if self.trial["practice"] else self.trial["index"] - 1
            stage = f"練習 {number} / 2" if self.trial["practice"] else f"本番 {number} / 4"
            label = f"{stage} — AI予測{'あり' if self.trial['forecast'] else 'なし'}"
            self.root.title(label + ("（自動確認・人の評価ではありません）" if study.automated else ""))
            self.heading_label.configure(text=label)
            self.intro_label.configure(text="3投で合計点を狙ってください。方向・パワー・回転を選び、滑走中はSpaceでスイープ。終了後に感想を記録します。")
            self.retry_button.configure(text="比較中はリトライなし", state="disabled")
            self.spin.set(0)
            self._input_changed()

        def launch(self):
            count = len(self.session.history)
            decision_time = time.perf_counter() - self.ready_since
            super().launch()
            if len(self.session.history) > count:
                self.decision_seconds.append(decision_time)

        def retry(self):
            return  # Keep the first attempt; interrupted games are explicitly recorded.

        def _autoplay_tick(self):
            if self.world.phase != "RESULT":
                super()._autoplay_tick()

        def tick(self):
            super().tick()
            if not self.closed and self.world.phase == "RESULT" and not self.captured:
                study.capture(self.session, time.perf_counter() - self.started, self.decision_seconds,
                              self.forecasts, self.errors)
                self.captured = True
                self.root.after(100, self.collect_feedback)

        def collect_feedback(self):
            if not self.closed and not self.feedback_open:
                self.feedback_open = True
                self.advance = feedback(self.root, study)
                self.close()

        def close(self):
            if self.closed:
                return
            if not self.captured:
                study.interrupt("Window closed before a complete game was recorded.")
            super().close()

    return StudyWindow()


def run_playtest(directory=None, seed=20261101, automated=False):
    import tkinter as tk
    from tkinter import messagebox
    study = PlaytestStudy(directory, seed, automated)
    print(f"Playtest directory: {study.directory}", flush=True)
    if not study.data["attempts"] and not automated:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("比較プレイの進め方", "練習2ゲーム＋本番4ゲームです（各3投、目安10〜15分）。\n\n"
                            "各ゲームで高い合計点を狙い、終了後に3つの感想を選んでください。\n"
                            "AI予測の表示あり・なしを比較します。回転とスイープは自由に使えます。\n"
                            "途中で閉じると中断として記録し、あとで再開できます。\n\n"
                            "表示される予測は『ここからスイープなし』の目安です。", parent=root)
        root.destroy()
    while study.next_trial is not None:
        if study.data["pending"] is not None:
            root = tk.Tk()
            root.withdraw()
            accepted = feedback(root, study)
            root.destroy()
            if not accepted:
                break
            continue
        record = study.start_attempt()
        window = make_window(study, record)
        window.root.mainloop()
        if not window.advance:
            break
    summary = summarize_study(study.directory)
    print(summary["text"], flush=True)
    if automated:
        if not summary["complete"] or summary["ui_errors"] or summary["display_failures"]:
            raise RuntimeError("Automated playtest check failed; inspect the study directory.")
        assert not summary["human_evidence"] and summary["mean_with_minus_without"] is None
    else:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("比較プレイの記録", ("比較プレイが完了しました。" if summary["complete"] else "途中までの記録を保存しました。") +
                            f"\n\n保存先：\n{study.directory}\n\n結果はsummary.mdに保存しました。", parent=root)
        root.destroy()
    return study.directory
