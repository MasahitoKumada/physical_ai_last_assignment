"""Native control window for the curling game and the earlier technical PoC.

Rendering and Tk stay on the main thread. A single CPU-only worker computes
forecasts in an independent physics world. The game uses estimated ice;
the legacy PoC uses known ice for diagnostics.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from PIL import Image, ImageTk

from physics import CurlingWorld, DT, ShotCommand, reference_forecast
from predictor import predict_board
from player_view import ice_hint, shot_outcome, shot_preview, render_player_panel

ROOT = Path(__file__).resolve().parents[1]


class PocWindow:
    def __init__(self, renderer_factory, smoke_seconds=None, session=None, autoplay=False,
                 prediction_enabled=True, record_path=None):
        self.session = session
        self.autoplay = autoplay
        self.prediction_enabled = prediction_enabled
        self.record_path = Path(record_path) if record_path else ROOT / "report/game_last_session.json"
        self.auto_index, self.auto_prepared = 0, False
        self.result_since = None
        self.ready_since = time.perf_counter()
        self.last_prediction_sim = -float("inf")
        self.turn_start_result = None
        self.last_outcome = None
        self.best_score = None
        self.last_hud_time = -float("inf")
        self.house_camera = False
        self.root = tk.Tk()
        self.root.title("ロボット・カーリング — 変わる氷を読み、3投で中心へ" if session else "ロボット・カーリング PoC — AI学習前の技術検証")
        self.root.geometry("1160x740")
        if session:
            self._style_game()
        self.world = session.world if session else CurlingWorld()
        self.renderer = renderer_factory(self.world, width=800, height=450)
        if session:
            self.renderer.camera.distance = 3.5
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.future_revision = -1
        self.revision = 0
        self.last_change = time.perf_counter() - 1
        self.last_tick = time.perf_counter()
        self.accumulator = 0.
        self.started = self.last_tick
        self.frames = 0
        self.forecasts = 0
        self.forecast_latencies = []
        self.errors = []
        self.closed = False
        self.sweeping = False
        self.smoke_seconds = smoke_seconds
        self.angle = tk.DoubleVar(value=0.)
        self.power = tk.DoubleVar(value=79.)
        self.spin = tk.IntVar(value=0)
        self.angle_text = tk.StringVar()
        self.power_text = tk.StringVar()
        self.status = tk.StringVar(value="方向とパワーを選んで投球してください。")
        self.prediction_status = tk.StringVar(value="予測を準備しています。")
        self.score_status = tk.StringVar(value="投球 0 / 3")
        self.ai_status = tk.StringVar(value="初投は標準的な氷を仮定します。")
        self.plan_status = tk.StringVar(value="この1投の行き先を予測")
        self.plan_detail = tk.StringVar(value="方向とパワーを変えて、止まる場所を確かめよう。")
        self._build_widgets()
        self._input_changed()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress-space>", lambda e: self._sweep(True))
        self.root.bind("<KeyRelease-space>", lambda e: self._sweep(False))
        self.root.bind("<FocusOut>", lambda e: self._sweep(False))
        self.root.bind_all("<ButtonRelease-1>", lambda e: self._sweep(False), add=True)
        self.root.after(20, self.tick)
        if smoke_seconds is not None and not autoplay:
            self.root.after(700, self.launch)
            self.root.after(1600, lambda: self._sweep(True))
            self.root.after(2100, lambda: self._sweep(False))
            self.root.after(round(smoke_seconds * 1000), self.close)
        elif autoplay:
            self.root.after(45000, self.close)

    def _style_game(self):
        self.root.configure(background="#0b1728")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#102238")
        style.configure("TLabel", background="#102238", foreground="#e9f2f8")
        style.configure("TButton", background="#203c57", foreground="#e9f2f8", padding=(8, 5))
        style.map("TButton", background=[("active", "#315574"), ("disabled", "#172b40")],
                  foreground=[("disabled", "#71879c")])
        style.configure("Throw.TButton", background="#62d9df", foreground="#082437", font=("Hiragino Sans", 14, "bold"))
        style.map("Throw.TButton", background=[("active", "#95edf0"), ("disabled", "#224251")],
                  foreground=[("disabled", "#7b9ba6")])
        style.configure("Nudge.TButton", padding=(3, 2))
        style.configure("TRadiobutton", background="#102238", foreground="#e9f2f8")
        style.map("TRadiobutton", background=[("active", "#1b3650")], foreground=[("disabled", "#71879c")])
        style.configure("Horizontal.TScale", background="#102238", troughcolor="#29435d")

    def _build_widgets(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, minsize=300)
        self.heading_label = ttk.Label(self.root, text="ROBOT CURLING  /  3投で、中心へ。" if self.session else "3投カーリング・技術検証PoC", font=("Hiragino Sans", 19))
        self.heading_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(12, 4))
        self.intro_label = ttk.Label(self.root, text="半透明の石が止まる目安。方向とパワーで中心を狙い、最後に残った3つの石でスコアを伸ばそう。" if self.session else "予測は既知の氷で計算しています。学習済みAIの接続は次の段階です。",
                                     foreground="#a9bfd1" if self.session else "#7d4b25")
        self.intro_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=16)
        self.view_frame = ttk.Frame(self.root, width=824, height=620)
        self.view_frame.grid(row=2, column=0, sticky="nw", pady=(10, 0))
        self.view_frame.grid_propagate(False)
        self.image_label = ttk.Label(self.view_frame, padding=0, borderwidth=0)
        self.image_label.place(x=10, y=0, width=800, height=450)
        self.hud_label = ttk.Label(self.view_frame, padding=0, borderwidth=0)
        self.hud_label.place(x=10, y=462, width=800, height=152)
        panel = ttk.Frame(self.root, padding=12)
        panel.grid(row=2, column=1, sticky="nsew")
        self._fixed_status_label(panel, self.score_status, font=("Hiragino Sans", 19), lines=1, pady=(0, 8))
        self.adjustments = []
        ttk.Label(panel, textvariable=self.angle_text).pack(anchor="w")
        self.angle_slider = ttk.Scale(panel, from_=-15, to=15, variable=self.angle, command=self._input_changed)
        self.angle_slider.pack(fill="x", pady=(4, 2))
        self._nudge_row(panel, self.angle, .5, -15., 15., "左へ +0.5°", "右へ −0.5°")
        ttk.Label(panel, textvariable=self.power_text).pack(anchor="w")
        self.power_slider = ttk.Scale(panel, from_=1, to=100, variable=self.power, command=self._input_changed)
        self.power_slider.pack(fill="x", pady=(4, 2))
        self._nudge_row(panel, self.power, 1., 1., 100., "強く +1", "弱く −1")
        ttk.Label(panel, text="回転").pack(anchor="w")
        rotation = ttk.Frame(panel)
        rotation.pack(fill="x", pady=(4, 8))
        self.radios = []
        for label, value in [("左", 1), ("なし", 0), ("右", -1)]:
            radio = ttk.Radiobutton(rotation, text=label, variable=self.spin, value=value, command=self._input_changed)
            radio.pack(side="left", padx=3)
            self.radios.append(radio)
        self.shoot_button = ttk.Button(panel, text="この狙いで投げる", command=self.launch, style="Throw.TButton" if self.session else "TButton")
        self.shoot_button.pack(fill="x", pady=4)
        self.sweep_button = ttk.Button(panel, text="押している間スイープ\n（またはSpace）")
        self.sweep_button.pack(fill="x", pady=4)
        self.sweep_button.bind("<ButtonPress-1>", lambda e: self._sweep(True))
        self.sweep_button.bind("<ButtonRelease-1>", lambda e: self._sweep(False))
        self.camera_button = ttk.Button(panel, text="ハウスを拡大して狙う", command=self.toggle_camera)
        self.camera_button.pack(fill="x", pady=4)
        self.retry_button = ttk.Button(panel, text="同じ条件でリトライ", command=self.retry)
        self.retry_button.pack(fill="x", pady=4)
        self._fixed_status_label(panel, self.plan_status, font=("Hiragino Sans", 16), wraplength=248, lines=1, pady=6)
        self._fixed_status_label(panel, self.plan_detail, wraplength=248, pady=2)
        if self.prediction_enabled:
            ttk.Label(panel, text="予測＝この1投後・スイープなし。\nかする衝突はズレに注意。", font=("Hiragino Sans", 11), wraplength=248).pack(anchor="w", pady=2)
        ttk.Label(self.root, textvariable=self.status, wraplength=1200).grid(row=3, column=0, columnspan=2,
                                                                          sticky="w", padx=16, pady=5)

    def _nudge_row(self, parent, variable, step, low, high, plus, minus):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 7))
        for text, delta in ((minus, -step), (plus, step)):
            button = ttk.Button(row, text=text, style="Nudge.TButton",
                                command=lambda change=delta: self._nudge(variable, change, low, high))
            button.pack(side="left", fill="x", expand=True, padx=2)
            self.adjustments.append(button)

    def _nudge(self, variable, delta, low, high):
        if self.world.phase == "READY":
            variable.set(max(low, min(high, variable.get() + delta)))
            self._input_changed()

    def _fixed_status_label(self, parent, variable, font="TkDefaultFont", wraplength=240, pady=0, lines=2):
        line_height = tkfont.Font(root=self.root, font=font).metrics("linespace")
        slot = ttk.Frame(parent, width=wraplength, height=lines * line_height + 4)
        slot.pack(fill="x", pady=pady)
        slot.pack_propagate(False)
        ttk.Label(slot, textvariable=variable, font=font, wraplength=wraplength).pack(anchor="nw")

    def _input_changed(self, _=None):
        self.revision += 1
        self.last_change = time.perf_counter()
        self.renderer.prediction = None
        self.renderer.predicted_trace = []
        self.last_hud_time = -float("inf")
        self.angle_text.set(f"方向：{self.angle.get():+.1f}°（＋が左）")
        self.power_text.set(f"パワー：{self.power.get():.0f} / 100")

    def _command(self):
        return ShotCommand(float(self.angle.get()), float(self.power.get()), int(self.spin.get()))

    def _sweep(self, enabled):
        updated = bool(enabled and self.world.phase in ("SLIDING", "SETTLING"))
        if updated != self.sweeping and self.session:
            self.revision += 1
            self.renderer.prediction, self.renderer.predicted_trace = None, []
            self.last_prediction_sim = -float("inf")
        self.sweeping = updated

    def launch(self):
        try:
            before = self.world.result()
            if self.session:
                self.session.launch(self._command())
                self.revision += 1
                self.last_prediction_sim = -float("inf")
            else:
                self.world.start_shot(self._command())
            self.turn_start_result = before
            self.last_hud_time = -float("inf")
            self.status.set("ロボットが保持・回転・解放します。滑走中はSpaceでスイープできます。")
        except Exception as error:
            self.status.set(str(error))
            self.errors.append(str(error))

    def retry(self):
        if self.session:
            self.session.reset()
        else:
            self.world.friction = .05
            self.world.reset()
        self.sweeping = False
        self.last_outcome = self.turn_start_result = None
        self.errors.clear()
        self._input_changed()
        self.last_prediction_sim = -float("inf")
        self.status.set("氷・盤面・AIのゲーム中補正を初期状態に戻しました。" if self.session else "氷と盤面を初期状態に戻しました。")

    def toggle_camera(self):
        self.house_camera = not self.house_camera
        self.renderer.camera.lookat[:] = [3., 0., .03] if self.house_camera else [2., 0., .06]
        self.renderer.camera.distance = 1.65 if self.house_camera else (3.5 if self.session else 4.8)
        self.renderer.camera.elevation = -89 if self.house_camera else -57
        self.camera_button.configure(text="リンク全体を見る" if self.house_camera else "ハウスを拡大して狙う")

    def _prediction_tick(self):
        if not self.prediction_enabled:
            self.renderer.prediction, self.renderer.predicted_trace = None, []
            self.prediction_status.set("AI予測なし\n実際の動きと配置を見て狙います。")
            return
        if self.future is not None and self.future.done():
            try:
                result, trace = self.future.result()
                self.forecasts += 1
                self.forecast_latencies.append(time.perf_counter() - self.future_started)
                age = self.world.data.time - getattr(self, "future_observed_time", self.world.data.time)
                if self.future_revision == self.revision and (not self.session or self.world.phase == "READY" or age < 1.):
                    self.renderer.prediction, self.renderer.predicted_trace = result, trace
                    self.prediction_status.set("ここからスイープしない場合の目安\nかする衝突では外れることもあります" if self.session else f"既知条件の予測：完了\n応答 {self.forecast_latencies[-1]:.2f}秒")
            except Exception as error:
                self.errors.append(str(error))
                self.prediction_status.set(f"予測エラー：{error}")
            self.future = None
        ready_due = self.world.phase == "READY" and self.renderer.prediction is None and time.perf_counter() - self.last_change > .2
        flight_due = self.session and self.world.phase == "SLIDING" and self.world.data.time - self.last_prediction_sim >= .4
        if self.future is None and (ready_due or flight_due):
            self.future_revision = self.revision
            self.future_started = time.perf_counter()
            self.future_observed_time = float(self.world.data.time)
            self.last_prediction_sim = self.future_observed_time
            if self.session:
                command = self._command() if self.world.phase == "READY" else None
                self.future = self.executor.submit(predict_board, self.world.observe_board(), self.session.estimator.estimate,
                                                   command, True)
            else:
                self.future = self.executor.submit(reference_forecast, self.world.snapshot(), self._command(),
                                                   self.world.friction, self.world.curl)
            self.prediction_status.set("止まる場所を読んでいます…\n予測はスイープなしの条件です")

    def _refresh_player_panel(self, force=False):
        if not force and time.perf_counter() - self.last_hud_time < .1:
            return
        actual = self.world.result()
        before = actual if self.world.phase == "READY" else self.turn_start_result or actual
        prediction = self.renderer.prediction if self.prediction_enabled and self.world.phase != "RESULT" else None
        preview = shot_preview(before, prediction)
        if self.world.phase == "RESULT":
            self.plan_status.set(f"FINISH  {actual['score']} 点")
            self.plan_detail.set("同じ氷で再挑戦。少し狙いを変えて、ベストを更新しよう。")
            self.prediction_status.set("各石の点数は最後の配置で確定します。")
        elif not self.prediction_enabled:
            self.plan_status.set("自分の感覚で狙おう")
            self.plan_detail.set("中心に近いほど高得点。置いた石を押し込む作戦も使えます。")
        elif preview:
            self.plan_status.set(f"予想合計 {preview['score']}点 ({preview['delta']:+d})")
            self.plan_detail.set(preview["detail"])
        else:
            self.plan_status.set("この1投の行き先を予測")
            self.plan_detail.set("方向とパワーを変えると、予想位置も変わります。")
        completed = self.session.completed if self.session else int(self.world.played.sum())
        if self.session and self.prediction_enabled:
            ice = ice_hint(self.session.estimator.estimate, self.session.model.prior_friction,
                           self.session.estimator.updates, completed, self.world.phase)
        else:
            ice = ("あなたの感覚で、氷を読もう", "滑り方と止まった場所を見て、次の一投を調整できます。")
        self.player_panel = render_player_panel(actual, prediction, before, completed=completed,
                                               phase=self.world.phase, ice=ice,
                                               outcome=self.last_outcome, best=self.best_score)
        self.hud_photo = ImageTk.PhotoImage(self.player_panel)
        self.hud_label.configure(image=self.hud_photo)
        self.last_hud_time = time.perf_counter()

    def save_player_preview(self, path):
        self._refresh_player_panel(force=True)
        preview = Image.new("RGB", (800, 614), "#0b1728")
        preview.paste(Image.fromarray(self.renderer.render()), (0, 0))
        preview.paste(self.player_panel, (0, 462))
        preview.save(path)

    def tick(self):
        if self.closed:
            return
        try:
            running = self.world.phase in ("DELIVERY", "SLIDING", "SETTLING")
            now = time.perf_counter()
            self.accumulator = min(.24, self.accumulator + now - self.last_tick) if running else 0.
            self.last_tick = now
            steps = int(self.accumulator / DT)
            self.accumulator -= steps * DT
            for _ in range(steps):
                if self.session:
                    self.session.step(self.sweeping)
                else:
                    self.world.step(self.sweeping)
            if running and self.world.phase in ("READY", "RESULT"):
                self.sweeping = False
                result = self.world.result()
                self.last_outcome = shot_outcome(self.turn_start_result or {"score": 0, "stones": []}, result, self.world.current)
                self.last_hud_time = -float("inf")
                if self.world.phase == "RESULT":
                    self.best_score = max(self.best_score or 0, result["score"])
                self.status.set(f"投球終了。暫定得点 {result['score']}点。" if self.world.phase == "READY"
                                else f"3投終了！ 最終得点 {result['score']}点。")
                if self.world.phase == "READY":
                    if not self.session:
                        self.world.friction = .05 * (1 - .05 * int(self.world.played.sum()))
                    self.ready_since = time.perf_counter()
                    self._input_changed()
                elif self.session:
                    self.revision += 1
                    self.renderer.prediction, self.renderer.predicted_trace = None, []
            if self.world.phase == "ERROR":
                if self.world.error not in self.errors:
                    self.errors.append(self.world.error)
                self.status.set(f"物理エラー：{self.world.error}。リトライできます。")
            enabled = self.world.phase == "READY"
            for control in [self.angle_slider, self.power_slider, self.shoot_button, *self.radios, *self.adjustments]:
                control.configure(state="normal" if enabled else "disabled")
            label = "最終" if self.world.phase == "RESULT" else "暫定"
            self.score_status.set(f"{int(self.world.played.sum())} / 3 投   {self.world.result()['score']} 点")
            if self.session:
                updates = self.session.estimator.updates
                self.ai_status.set(f"石の動きから氷の推定を更新：{updates}回" if updates else "初投は標準的な氷を仮定します。")
            self._prediction_tick()
            self._refresh_player_panel()
            self.photo = ImageTk.PhotoImage(Image.fromarray(self.renderer.render()))
            self.image_label.configure(image=self.photo)
            self.frames += 1
            if self.autoplay:
                self._autoplay_tick()
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")
            self.status.set(f"エラー：{error}")
        if not self.closed:
            self.root.after(10, self.tick)

    def _autoplay_tick(self):
        from game import DEMO_COMMANDS
        if self.world.phase == "READY" and self.auto_index < 3:
            if not self.auto_prepared:
                command = DEMO_COMMANDS[self.auto_index]
                self.angle.set(command.angle)
                self.power.set(command.power)
                self.spin.set(command.spin)
                self._input_changed()
                self.auto_prepared = True
                self.ready_since = time.perf_counter()
            if (not self.prediction_enabled or self.renderer.prediction is not None) and time.perf_counter() - self.ready_since > .8:
                if self.session and self.smoke_seconds is not None and self.auto_index == 1:
                    self.save_player_preview(ROOT / "report/game_aim_preview.png")
                    self.toggle_camera()
                    self.save_player_preview(ROOT / "report/game_house_preview.png")
                    self.toggle_camera()
                self.launch()
                self.auto_index += 1
                self.auto_prepared = False
        if self.world.release is not None and self.world.phase == "SLIDING":
            elapsed = self.world.data.time - self.world.release
            self._sweep(self.auto_index == 1 and .25 <= elapsed < .45)
        if self.world.phase == "RESULT":
            if self.result_since is None:
                self.result_since = time.perf_counter()
            elif time.perf_counter() - self.result_since > 1.:
                self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.session:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self.session.save_record(self.record_path)
        if self.smoke_seconds is not None:
            report = ROOT / "report"
            report.mkdir(exist_ok=True)
            prefix = "game" if self.session else "poc"
            Image.fromarray(self.renderer.render()).save(report / f"{prefix}_gui_preview.png")
            if self.session:
                self.save_player_preview(report / "game_player_preview.png")
            elapsed = time.perf_counter() - self.started
            clipped = []
            def inspect_layout(widget):
                for child in widget.winfo_children():
                    if child.winfo_ismapped() and not child.winfo_children():
                        x = child.winfo_rootx() - self.root.winfo_rootx()
                        y = child.winfo_rooty() - self.root.winfo_rooty()
                        if x < 0 or y < 0 or x + child.winfo_width() > self.root.winfo_width() or y + child.winfo_height() > self.root.winfo_height():
                            clipped.append({"widget": child.winfo_class(), "x": x, "y": y,
                                            "width": child.winfo_width(), "height": child.winfo_height()})
                    inspect_layout(child)
            inspect_layout(self.root)
            result = {"window_created": True, "frames_rendered": self.frames,
                      "wall_seconds": elapsed, "average_frames_per_second": self.frames / elapsed,
                      "simulation_seconds": float(self.world.data.time), "forecasts_completed": self.forecasts,
                      "forecast_response_seconds": self.forecast_latencies,
                      "errors": self.errors, "phase_at_close": self.world.phase,
                      "learned_ai": self.session is not None,
                      "shots_completed": self.session.completed if self.session else int(self.world.played.sum()),
                      "estimate_updates": self.session.estimator.updates if self.session else 0,
                      "final_score": self.world.result()["score"], "clipped_widgets": clipped}
            (report / f"{prefix}_gui_results.json").write_text(json.dumps(result, indent=2) + "\n")
            if self.session:
                self.session.save_record(report / "game_gui_replay.json")
            print(json.dumps(result, indent=2), flush=True)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.renderer.close()
        for callback in self.root.tk.splitlist(self.root.tk.call("after", "info")):
            self.root.after_cancel(callback)
        self.root.destroy()


def run_poc_ui(renderer_factory, smoke_seconds=None):
    app = PocWindow(renderer_factory, smoke_seconds)
    app.root.mainloop()


def run_game_ui(renderer_factory, session, smoke=False):
    app = PocWindow(renderer_factory, smoke_seconds=45. if smoke else None, session=session, autoplay=smoke)
    app.root.mainloop()
    if smoke:
        result = json.loads((ROOT / "report/game_gui_results.json").read_text())
        if result["errors"] or result["clipped_widgets"] or result["phase_at_close"] != "RESULT":
            raise RuntimeError("GUI acceptance check failed; see report/game_gui_results.json.")
