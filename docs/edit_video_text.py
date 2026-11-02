"""ゲーム映像に、表紙・制作意図・学習式・予測の使い方を付ける。

再生成時は先に python src/game.py video を実行する。
ゲームの入力・物理計算・評価結果には変更を加えない。
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from player_view import fit
from poc import font_path

OLD_HEADING = 'できたことと、残る限界'
NEW_HEADING = '今回できたことと今後の課題'
CLOSING = '人が狙い、ロボットが投げ、AIが氷を読む。'
REVISION = 'cover_concept_equations_roles_v3'
BG = '#10243b'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def card(title, lines, chart=None):
    image = Image.new('RGB', (1280, 720), BG)
    draw = ImageDraw.Draw(image)
    fit(draw, title, 55, 45, 1170, 42, 'white')
    y = 210
    if chart is not None:
        picture = chart.copy().convert('RGB')
        picture.thumbnail((1120, 425))
        image.paste(picture, ((1280-picture.width)//2, 140))
        y = 580
    for line in lines:
        fit(draw, line, 60, y, 1160, 28, '#d3e5f0')
        y += 55
    fit(draw, 'Physical AI 応用1 最終課題 / bonsai2026', 60, 680, 1160, 20, '#88afc5')
    return image


def cover():
    image = Image.new('RGB', (1280, 720), BG)
    draw = ImageDraw.Draw(image)
    fit(draw, 'Physical AI 応用1 最終課題', 55, 28, 1170, 22, '#b3cbdc')
    fit(draw, 'ロボット・カーリング', 55, 78, 1170, 48, 'white')
    fit(draw, 'アカウント名：bonsai2026', 55, 148, 1170, 27, 'white')
    fit(draw, 'ゲームの概要', 55, 205, 1170, 27, '#62d9df')
    fit(draw, '方向と強さは人が選び、ロボットが投球します。', 55, 250, 1170, 28, '#d3e5f0')
    fit(draw, '開発したもの', 55, 325, 570, 27, '#62d9df')
    for index, line in enumerate(['ロボットで3投する1人用ゲーム',
                                  '石の動きから、AIが氷の状態を推定',
                                  '衝突した後の停止位置も予測']):
        fit(draw, line, 55, 385+index*60, 570, 27, 'white')
    with Image.open(ROOT / 'report/game_player_preview.png') as picture:
        picture = picture.convert('RGB')
        picture.thumbnail((560, 430))
        image.paste(picture, (660, 255))
    fit(draw, 'ゲーム画面', 660, 692, 560, 17, '#b3cbdc')
    return image


def learning():
    import io
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/curling-learning-mpl')
    from matplotlib import rc_context
    from matplotlib.mathtext import math_to_image
    image = card('AIが学習したのは、氷を推定する2つの式', [])
    draw = ImageDraw.Draw(image)
    for offset, label, equation in [
            (205, '摩擦の推定', r'$\hat{\mu}=a_0+a_1d$'),
            (305, '曲がりやすさの推定', r'$\hat{\kappa}=b_0+b_1c$')]:
        fit(draw, label, 60, offset+12, 500, 29, '#d3e5f0')
        buffer = io.BytesIO()
        with rc_context({'savefig.facecolor': BG, 'figure.facecolor': BG}):
            math_to_image(equation, buffer, dpi=260, format='png', color='white')
        buffer.seek(0)
        formula = Image.open(buffer).convert('RGBA')
        formula.thumbnail((550, 70))
        image.paste(formula, (650, offset), formula)
    lines = [
        'd：減速の指標　c：曲がりの指標（回転・速度などの影響を補正）',
        '正解の氷の値に近づくよう、a₀・a₁・b₀・b₁を学習しました。',
        '学習は189投、設定の調整には別の72投を使いました。',
    ]
    for index, line in enumerate(lines):
        fit(draw, line, 60, 425+index*58, 1160, 25, '#d3e5f0')
    draw.rectangle((0, 635, 1279, 719), fill=BG)
    fit(draw, '式は回帰の基本形です。ゲーム中は4つの係数を固定し、新しい動きから氷を推定します。',
        60, 655, 1160, 20, '#b3cbdc')
    return image


def usage():
    image = card('検証を通じて決めた、予測の使い方', [])
    draw = ImageDraw.Draw(image)
    fit(draw, '停止位置は物理計算で求め、氷の値の決め方を変えます。', 60, 140, 1160, 28, '#d3e5f0')
    rows = [['段階', '計算に使う氷の値', '予測の役割'],
            ['1投目の投球前', '標準値', '大まかな目安'],
            ['1投目の滑走中', 'AIが動きから推定', '停止位置を読み直す'],
            ['2・3投目の投球前', '前投の推定値', '方向と強さの狙いを調整']]
    for index, row in enumerate(rows):
        y = 235 + index*75
        draw.rectangle((55, y, 1225, y+75), fill='#276378' if index==0 else '#19334e' if index%2 else '#1d3d57')
        x = 55
        for value, width in zip(row, [320, 330, 520]):
            fit(draw, value, x+18, y+24, width-36, 25, 'white')
            x += width
    fit(draw, '初投前は平均約29cmずれ、精密な狙いには使いにくい予測でした。',
        60, 568, 1160, 25, '#ffd18a')
    fit(draw, '次の投球の氷の変化は、投げるまで分かりません。', 60, 625, 1160, 24, '#d3e5f0')
    return image


def ending():
    image = card(NEW_HEADING, [])
    draw = ImageDraw.Draw(image)
    fit(draw, '今回できたこと', 60, 165, 1160, 29, '#62d9df')
    for index, line in enumerate([
        '3投・回転・スイープ・衝突を含むゲームに、氷の推定と停止位置予測を組み込みました。',
        '精度を比較し、初投と2・3投目で予測をどう使うかを整理しました。',
    ]):
        fit(draw, line, 60, 220+index*55, 1160, 27, '#d3e5f0')
    fit(draw, '今後の課題', 60, 380, 1160, 29, '#ffd18a')
    for index, line in enumerate([
        '初投前の情報不足と、観測誤差で衝突予測が大きく外れる点への対処が必要です。',
        '学習が役立つ条件と、得点や面白さへの効果には、条件・人数を増やした検証が必要です。',
        '実際の氷や実機で使えるかも未検証です。',
    ]):
        fit(draw, line, 60, 435+index*55, 1160, 27, '#d3e5f0')
    return image


def comparison_slide(path, title, subtitle, footer):
    with Image.open(ROOT / 'report' / path) as source:
        image = source.convert('RGB')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1279, 129), fill=BG)
    fit(draw, title, 45, 27, 1190, 38, 'white')
    fit(draw, subtitle, 45, 86, 1190, 23, '#b3cbdc')
    draw.rectangle((0, 658, 1279, 719), fill=BG)
    fit(draw, footer, 45, 668, 1190, 19, '#b3cbdc')
    return image


def evaluation_chart(method_comparison=False):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/curling-learning-mpl')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.font_manager import FontProperties
    fp = FontProperties(fname=font_path())
    figure = Figure(figsize=(10.5, 3.8), dpi=160, layout='constrained')
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    if method_comparison:
        data = json.loads((ROOT / 'report/game_method_comparison.json').read_text())
        groups = ['clean', 'small_noise', 'larger_noise']
        x = np.arange(3)
        for offset, method, label, color in [(-.25, 'fixed', '標準値', '#bc7452'),
                                             (0., 'analytic', '物理式', '#9878bc'),
                                             (.25, 'learned', '学習モデル', '#23899d')]:
            for axis, key in zip(axes, ['mean_stop_error_m', 'max_endpoint_error_m']):
                values = [data['summaries'][g]['overall'][method][key]*100 for g in groups]
                bars = axis.bar(x+offset, values, .25, label=label, color=color)
                axis.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
                axis.margins(y=.25)
        for axis, title in zip(axes, ['場内で止まった石の平均のずれ', '最も大きく外れた石のずれ']):
            axis.set_title(title, fontproperties=fp, fontsize=12)
            axis.set_xticks(x, ['誤差なし', '小さい誤差', '大きい誤差'], fontproperties=fp, fontsize=10)
            axis.set_ylabel('停止位置のずれ（cm）', fontproperties=fp, fontsize=10)
    else:
        data = json.loads((ROOT / 'report/game_evaluation.json').read_text())
        for axis, timing, title in zip(axes, ['pre_shot', 'in_flight'], ['投球前', '石を離して1秒後']):
            for method, label, color in [('fixed', '標準の氷で固定', '#bc7452'), ('learned', '観測から補正', '#23899d')]:
                values = [data['summaries'][timing]['by_turn'][str(t)][method]['mean_stop_error_m']*100 for t in [1,2,3]]
                axis.plot([1,2,3], values, 'o-', label=label, color=color)
            axis.set_title(title, fontproperties=fp, fontsize=12)
            axis.set_xticks([1,2,3], ['1投目', '2投目', '3投目'], fontproperties=fp, fontsize=10)
            axis.set_ylabel('停止位置の平均のずれ（cm）', fontproperties=fp, fontsize=10)
    legend_font = FontProperties(fname=font_path(), size=9)
    axes[0].legend(prop=legend_font)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())).convert('RGB')


def main():
    video = ROOT / 'video/game_demo.mp4'
    metadata_path = ROOT / 'report/game_video_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    chapters = metadata['chapters']
    if metadata.get('text_edits', {}).get('revision') == REVISION:
        print('Video text already updated.')
        return
    fps = metadata['fps']
    ranges = {c['title']: (round(c['start_seconds']*fps),
              round(chapters[i+1]['start_seconds']*fps) if i+1<len(chapters) else metadata['frames'])
              for i,c in enumerate(chapters)}
    review = ROOT / 'report/submission_review'
    review.mkdir(exist_ok=True)
    plan = []
    def still(title, picture, seconds, preview=None):
        plan.append((title, round(seconds*fps), picture, None))
        if preview:
            picture.save(review / preview)
    def shot(turn):
        start, stop = ranges[f'{turn}投目']
        plan.append((f'{turn}投目', stop-start, None, (start, turn)))
    still('表紙：ロボット・カーリング', cover(), 10, 'video_cover.png')
    still('制作の目的', card('カーリングの面白さを知ってもらうために', [
        'AIの予測を手がかりに、狙いを試せるゲームを目指しました。',
        '中心へ寄せるか、自分の石を押し込むか。',
        '狙いと結果を比べながら、考えられるようにしました。']), 10, 'video_concept.png')
    still('ゲームのルール', card('3投後の配置で、点数が決まります', [
        '的の円（ハウス）の中心は100点。外周にかかれば最低1点です。',
        '置いた石を押し込んで、点数を伸ばすこともできます。']), 8)
    shot(1)
    still('AIが学習した2つの式', learning(), 18, 'video_learning.png')
    still('投球の段階ごとの予測の使い方', usage(), 16, 'video_usage.png')
    shot(2)
    shot(3)
    still('3投終了：合計165点', card('3投終了：合計165点', [
        '後の投球で動いた石も、最後の位置で採点します。',
        '同じ氷で狙いを変え、もう一度プレイできます。']), 6)
    still('2投目：投球前の予測と結果', comparison_slide('game_comparison_turn_2.png',
        '2投目：投げる前の予測と実際の位置',
        '投げる前の推定を固定 / 同じ盤面・方向・強さ / その後のスイープなし',
        'この場面では標準値の方が正確でした。前投の氷の推定が、次の氷に合わない例です。'), 10)
    still('学習に使っていない氷での衝突予測', comparison_slide('game_prediction_example.png',
        '学習に使っていない氷で、衝突後の位置を予測',
        '衝突前の0.1〜0.7秒の動きから氷を推定 / その後のスイープなし',
        '既存テストの最初の正面衝突例です。精度は次の集計で全体を確認します。'), 9)
    pre = json.loads((ROOT/'report/game_evaluation.json').read_text())['summaries']['pre_shot']['by_turn']
    error_line = '投球前の平均のずれ：' + '、'.join(
        f"{t}投目 {pre[str(t)]['fixed']['mean_stop_error_m']*100:.1f}→{pre[str(t)]['learned']['mean_stop_error_m']*100:.1f} cm" for t in [2,3])
    still('24ゲーム・72投の予測精度', card('24ゲーム・72投で、予測のずれを比較', [error_line,
        '同じ操作で比較した結果です。人の得点が上がるかは別に確かめる必要があります。'], evaluation_chart()), 10)
    values = json.loads((ROOT/'report/game_method_comparison.json').read_text())['summaries']['small_noise']['overall']
    line = '小さな観測誤差を加えたとき：' + '、'.join(
        f"{label} {values[m]['mean_stop_error_m']*100:.2f} cm" for m,label in [('fixed','標準値'),('analytic','物理式'),('learned','学習')])
    still('物理式と学習モデルの比較', card('物理式だけで求める方法とも比べました', [line,
        '今回の条件では、物理式と学習モデルの精度はほぼ同じでした。'], evaluation_chart(True)), 14)
    still('予測が外れた例', comparison_slide('game_prediction_failure.png',
        'かすり衝突で、予測が大きく外れた例',
        '観測に大きな誤差を加えた条件 / 石を離して0.7秒後に予測',
        '大きな観測誤差を加えた96投のうち、最大のずれが出た例を選んでいます。'), 9)
    still(NEW_HEADING, ending(), 20, 'video_last_slide.png')
    expected = sum(row[1] for row in plan)
    if not 60 <= expected/fps <= 180:
        raise ValueError('Planned video is outside the assignment duration range.')
    if '--preview-only' in sys.argv:
        print(json.dumps({'planned_frames':expected,'seconds':expected/fps,'slides':[r[0] for r in plan]},ensure_ascii=False,indent=2))
        return
    before_hash = digest(video)
    shutil.copy2(video, review / 'video_before_final_edit.mp4')
    (review/'video_before_final_edit_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    edited = video.with_name('game_demo.edited.mp4')
    reader = imageio.get_reader(video)
    writer = imageio.get_writer(edited, fps=fps, codec='libx264', quality=8, macro_block_size=16,
                                ffmpeg_log_level='error', ffmpeg_params=['-pix_fmt', 'yuv420p'])
    count, final_chapters = 0, []
    try:
        for title, length, picture, original in plan:
            final_chapters.append({'title':title,'start_seconds':count/fps})
            for offset in range(length):
                if original is None:
                    frame = np.asarray(picture)
                else:
                    start, turn = original
                    frame = reader.get_data(start+offset)
                    if offset < 4*fps or offset < length-3*fps:
                        captions = {1:'1投目：まず中心へ。投球前の予測は大まかな目安です。',
                                    2:'2投目：前の石に当て、中心へ押し込みます。',
                                    3:'3投目：残った配置を見て、合計点を伸ばします。'}
                        if offset >= 4*fps:
                            captions = {1:'投球中の動きも、予測の更新に使います。',
                                        2:'石同士がぶつかった後の位置も、予測しています。',
                                        3:'この石が止まると、最後の配置で点数が決まります。'}
                        image = Image.fromarray(frame)
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0,640,1279,683),fill=BG)
                        fit(draw,captions[turn],25,646,1230,27,'white')
                        frame=np.asarray(image)
                writer.append_data(frame)
                count += 1
            print(f"video: {count/fps:.2f}s / {expected/fps:.2f}s",flush=True)
    finally:
        reader.close()
        writer.close()
    if count != expected or not 60 <= count / fps <= 180:
        raise RuntimeError('Unexpected output frame count or duration.')

    verification = imageio.get_reader(edited)
    info = verification.get_meta_data()
    decoded = 0
    for index, frame in enumerate(verification):
        decoded += 1
        if index == count - 1:
            Image.fromarray(frame).save(review / 'video_final_frame.png')
    verification.close()
    if decoded != expected or list(info['size']) != [1280, 720] or info['fps'] != fps:
        raise RuntimeError('Output video verification failed.')
    edited.replace(video)
    metadata['chapters'] = final_chapters
    metadata.update(frames=count, duration_seconds=count/fps)
    metadata['text_edits'] = {
        'source_video_sha256': before_hash,
        'revision': REVISION,
        'heading_before': OLD_HEADING,
        'heading_after': NEW_HEADING,
        'removed_final_slide': CLOSING,
        'cover_account': 'bonsai2026',
        'learned_equations': ['mu_hat = a0 + a1*d', 'kappa_hat = b0 + b1*c'],
        'prediction_roles': 'Stopping positions always use physics; ice uses the prior before the first throw and observation-based estimates afterwards',
        'game_inputs_and_results_changed': False,
        'reproduce': 'python src/game.py video; python docs/edit_video_text.py',
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    validation = {'frames_decoded': decoded, 'duration_seconds': count/fps, 'fps': info['fps'],
                  'size': list(info['size']), 'codec': info['codec'], 'bytes': video.stat().st_size,
                  'sha256': digest(video)}
    (ROOT / 'report/game_video_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    shutil.copy2(video, ROOT / 'submission/PAI最終課題_bonsai2026.mp4')
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
