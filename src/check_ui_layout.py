"""Desktop regression check: status changes must not move the rendered rink."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from PIL import Image, ImageTk

from game import GameSession, ROOT
from poc import PocRenderer
from ui import PocWindow


def check_layout():
    rows, hidden, clipped = [], [], []
    with tempfile.TemporaryDirectory(prefix="curling-layout-") as directory:
        for enabled in (False, True):
            app = PocWindow(PocRenderer, session=GameSession(), prediction_enabled=enabled,
                            record_path=Path(directory) / "unused_session.json")
            try:
                for callback in app.root.tk.splitlist(app.root.tk.call("after", "info")):
                    app.root.after_cancel(callback)
                app.photo = ImageTk.PhotoImage(Image.fromarray(app.renderer.render()))
                app.image_label.configure(image=app.photo)
                for score in ("0 / 3 投   0 点", "3 / 3 投   140 点"):
                    for prediction in ("予測を準備しています。", "別の世界で先読み計算中…",
                                       "AI予測：更新済み\nここからスイープしない場合"):
                        for ai in ("初投は標準的な氷を仮定します。", "石の動きから氷の推定を更新：9回",
                                   "石の動きから氷の推定を更新：10回"):
                            app.score_status.set(score)
                            app.prediction_status.set(prediction if enabled else "AI予測なし\n実際の動きと配置を見て狙います。")
                            app.ai_status.set(ai)
                            app.plan_status.set("予想合計 300点 (-300)")
                            app.plan_detail.set("1番が100点下がる見込み。配置に注意")
                            app.root.update_idletasks()
                            w = app.image_label
                            rows.append({"prediction_enabled": enabled, "score": score, "prediction": prediction,
                                         "ai": ai, "image": [w.winfo_x(), w.winfo_y(), w.winfo_width(), w.winfo_height()]})
                def inspect(widget):
                    for child in widget.winfo_children():
                        if not child.winfo_children():
                            if not child.winfo_ismapped():
                                hidden.append(str(child))
                            if child.winfo_class() == "TLabel" and child.winfo_reqheight() > child.winfo_height():
                                clipped.append({"widget": str(child), "required_height": child.winfo_reqheight(),
                                                "available_height": child.winfo_height()})
                            x = child.winfo_rootx() - app.root.winfo_rootx()
                            y = child.winfo_rooty() - app.root.winfo_rooty()
                            if x < 0 or y < 0 or x + child.winfo_width() > app.root.winfo_width() or y + child.winfo_height() > app.root.winfo_height():
                                clipped.append({"widget": str(child), "x": x, "y": y,
                                                "width": child.winfo_width(), "height": child.winfo_height()})
                        inspect(child)
                inspect(app.root)
            finally:
                app.close()
    rectangles = {tuple(row["image"]) for row in rows}
    result = {"passed": len(rectangles) == 1 and not hidden and not clipped,
              "states_checked": len(rows), "distinct_image_rectangles": [list(r) for r in sorted(rectangles)],
              "hidden_widgets": hidden, "clipped_widgets": clipped, "rows": rows,
              "ui_sha256": hashlib.sha256((ROOT / "src/ui.py").read_bytes()).hexdigest()}
    (ROOT / "report/game_layout_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise RuntimeError("The rink moved or a control was hidden/clipped.")
    return result


if __name__ == "__main__":
    check_layout()
