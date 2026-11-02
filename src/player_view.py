"""Player-facing feedback, derived only from visible results and ice estimates."""
from __future__ import annotations

from functools import lru_cache
import math

from PIL import Image, ImageDraw, ImageFont

STONE_COLORS = ("#ffba49", "#36d6df", "#d89aff")
BACKGROUND = "#102238"
MUTED = "#a9bfd1"


def ice_hint(estimate, prior_friction, updates, completed, phase):
    if not updates:
        if completed:
            return "AIの読みは、まだ標準のまま", "十分な滑走の観測が揃っていません。"
        return "まずは1投、氷の感触をつかもう", "AIはまだ標準的な氷で予測しています。"
    ratio = estimate.friction / prior_friction
    surface = "滑りやすそう" if ratio < .97 else "止まりやすそう" if ratio > 1.03 else "標準に近そう"
    headline = f"AIの氷読み：{surface}"
    if phase == "RESULT":
        detail = "3投の滑り方から読んだ氷です。同じ氷で再挑戦できます。"
    else:
        detail = "前の投球の読みです。次の氷の変化は投げて確かめます。" if phase == "READY" and completed else "石の滑り方を観測して、止まる場所を読み直しています。"
    return headline, detail


def shot_preview(before, prediction):
    """Compare whole-board scores, retaining decreases and removed stones."""
    if prediction is None:
        return None
    old = {stone["id"]: stone for stone in before["stones"]}
    new = {stone["id"]: stone for stone in prediction["stones"]}
    shifts = [(stone_id, new[stone_id]["score"] - stone["score"])
              for stone_id, stone in old.items() if stone_id in new]
    gains = [(stone_id, delta) for stone_id, delta in shifts if delta > 0]
    losses = [(stone_id, delta) for stone_id, delta in shifts if delta < 0]
    outgoing = [stone_id for stone_id, stone in new.items()
                if not stone["alive"] and (stone_id not in old or old[stone_id]["alive"])]
    if outgoing:
        detail = f"{outgoing[0] + 1}番の石は場外になりそう"
    elif losses:
        stone_id, delta = min(losses, key=lambda value: value[1])
        detail = f"{stone_id + 1}番が{abs(delta)}点下がる見込み。配置に注意"
    elif gains:
        stone_id, delta = max(gains, key=lambda value: value[1])
        detail = f"{stone_id + 1}番も押し込んで、さらに{delta}点の見込み"
    else:
        detail = "半透明の石が止まる場所の目安です"
    return {"score": prediction["score"], "delta": prediction["score"] - before["score"], "detail": detail}


def shot_outcome(before, after, stone_id):
    stone = next(stone for stone in after["stones"] if stone["id"] == stone_id)
    distance_cm = math.dist(stone["position"][:2], (3., 0.)) * 100 if stone["alive"] else None
    delta = after["score"] - before["score"]
    if delta < 0:
        title = "配置が動いた。次の作戦へ"
    elif stone["alive"] and distance_cm <= 7.:
        title = "中心にピタリ！"
    elif stone["score"] >= 60:
        title = "いい位置に止まった！"
    elif stone["score"] > 0:
        title = "ハウスイン！"
    elif delta > 0:
        title = "押し込みで加点！"
    elif not stone["alive"]:
        title = "場外。次はリンクの中へ"
    else:
        title = "あと少し、中心へ寄せよう"
    detail = "今回の石は場外です" if distance_cm is None else f"今回の石：中心まで {distance_cm:.0f} cm・{stone['score']} 点"
    return {"title": title, "detail": detail, "delta": delta,
            "distance_cm": distance_cm, "score": after["score"]}


@lru_cache(maxsize=16)
def font(size):
    from poc import font_path
    return ImageFont.truetype(font_path(), size)


def fit(draw, text, x, y, width, size, color):
    while size > 11 and draw.textlength(text, font=font(size)) > width:
        size -= 1
    draw.text((x, y), text, font=font(size), fill=color)


def render_player_panel(actual, prediction, before, *, completed, phase, ice, outcome=None, best=None):
    """The same fixed-size panel is used in the GUI and local previews."""
    image = Image.new("RGB", (800, 152), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for index in range(3):
        left = 10 + index * 108
        draw.rounded_rectangle((left, 9, left + 100, 94), radius=10, fill="#1a344f")
        draw.ellipse((left + 8, 18, left + 25, 35), fill=STONE_COLORS[index])
        draw.text((left + 32, 16), f"{index + 1}番", font=font(15), fill="white")
        stone = next((s for s in actual["stones"] if s["id"] == index), None)
        text = "未投球" if stone is None else "場外" if not stone["alive"] else f"{stone['score']} 点"
        fit(draw, text, left + 10, 41, 85, 21, "white")
        future = next((s for s in prediction["stones"] if s["id"] == index), None) if prediction else None
        if future:
            fit(draw, f"予測 {future['score']}点" if future["alive"] else "予測：場外", left + 10, 72, 85, 12, MUTED)
    if phase == "RESULT":
        title = f"FINISH  {actual['score']} 点"
        subtitle = f"次はベスト更新を狙おう（現在 {best} 点）。" if best is not None else "同じ氷でもう一度。狙いを少し変えてみよう。"
        detail = outcome["detail"] if outcome else "最後の配置で3個の合計点が決まります。"
    elif phase == "READY" and completed and outcome:
        title = f"{outcome['title']}  {outcome['delta']:+d} 点"
        subtitle, detail = outcome["detail"], "次は直接寄せる？ 置いた石を押し込む？"
    elif phase in ("SLIDING", "SETTLING"):
        title, subtitle, detail = "狙った場所へ、あと少し。", "Spaceを押すと、もう少し先へ滑ります。", "点数は止まってから確定。次の一投で配置は変わります。"
    elif phase == "DELIVERY":
        title, subtitle, detail = "ロボットが、あなたの狙いを実行。", "加速して、回転をつけて、リリース。", "滑り始めたらSpaceでスイープできます。"
    else:
        title, subtitle, detail = "3投で、中心へ。", "方向とパワーで狙い、投球ボタンで勝負。", "中心に近いほど高得点。外周にかかれば1点。"
    fit(draw, title, 348, 12, 438, 22, "#ffcf72")
    fit(draw, subtitle, 348, 47, 438, 15, "white")
    fit(draw, detail, 348, 73, 438, 13, MUTED)
    draw.line((12, 104, 788, 104), fill="#31506c")
    fit(draw, ice[0], 13, 114, 320, 14, "#74dfe4")
    fit(draw, ice[1], 348, 115, 438, 12, MUTED)
    return image
