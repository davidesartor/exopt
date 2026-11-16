"""exo-review: an experiment's static review page, plotly included, so it opens offline."""

from html import escape
from pathlib import Path
import argparse
import json
import time
import webbrowser

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from exopt import trials
from exopt.figures import (CLICKABLE, DARK, LIGHT, PAGE_CSS, TRIAL, WEB, animation_figure, estimate_figure, loss_figure, portrait_figure,
                           power_figure, profile_figure, profiles_figure, review_headline, trial_summary)


# figures are built light; the page swaps each light palette color for its dark counterpart
DARK_COLORS = {
    light: dark
    for light_value, dark_value in zip(LIGHT, DARK)
    for light, dark in zip(*((v,) if isinstance(v, str) else v for v in (light_value, dark_value)))
    if light != dark
}


def review_html(exp_dir: str) -> str:
    """The review page of the experiment in exp_dir, plotly included, so it opens offline."""
    history = trials.load(exp_dir)
    name = Path(exp_dir).resolve().name
    to_json = lambda figure: go.Figure(figure).to_json()

    optimization = (loss_figure, profiles_figure)
    trial_pages = [
        dict(index=t.index, summary=trial_summary(t), figures=dict(zip(TRIAL, map(to_json, (
            profile_figure("profile and its atoms", t.profile, t.samples, "measured_torque", atoms=True,
                           power_assist=t.power_assist),
            power_figure(t),
            estimate_figure("convergence of the estimate", t.samples, t.settings),
            portrait_figure(t),
        )))))
        for t in history
    ]
    best = min(range(len(history)), key=lambda k: history[k].loss, default=0)
    data = dict(
        optimization={key: to_json(figure(history, None)) for key, figure in zip(CLICKABLE, optimization)},
        trials=trial_pages, best=best,
    )
    # keep "</script>" inside the data from closing its tag
    embed = lambda value: json.dumps(value).replace("</", "<\\/")

    headline = review_headline(history) if history else "no trials"
    graphs = lambda ids, prefix="": "".join(f'<div class="card"><div id="{prefix}{i}"></div></div>' for i in ids)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(name)} review</title>
<style>{PAGE_CSS}
select {{ font: inherit; font-size: 13px; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--axis);
  background: var(--surface); color: var(--ink); }}</style>
<script>{get_plotlyjs()}</script>
</head><body><div class="page">
<header><h1>{escape(name)}</h1>
<span>{escape(headline)} · written {time.strftime("%Y-%m-%d %H:%M")}</span>
</header>
<h2>optimization</h2>
<div class="grid">{graphs(CLICKABLE)}</div>
<h2>trial</h2>
<div class="controls">
<span id="summary"></span></div>
<div class="grid">{graphs(TRIAL[:2], "trial-")}</div><div class="grid">{graphs(TRIAL[2:], "trial-")}</div>
</div>
<script type="application/json" id="review-data">{embed(data)}</script>
<script type="application/json" id="dark-colors">{embed(DARK_COLORS)}</script>
<script>{(WEB / "review.js").read_text()}</script>
</body></html>
"""


def write_review(exp_dir: str) -> Path:
    """Write <exp_dir>/dashboard.html."""
    out = Path(exp_dir) / "dashboard.html"
    out.write_text(review_html(exp_dir))
    return out


def main():
    parser = argparse.ArgumentParser(description="exo-review: the static review page of an experiment.")
    parser.add_argument("exp_dir", help="experiment folder")
    parser.add_argument("--animation", action="store_true", help="export the gait animation instead")
    parser.add_argument("--swing", type=float, default=45.0, help="animation max swing angle (deg)")
    parser.add_argument("--fps", type=float, default=25.0, help="animation frame rate")
    args = parser.parse_args()
    if not args.animation:
        print(f"Wrote {write_review(args.exp_dir)}")
        return
    out = Path(args.exp_dir) / "animation.html"
    animation_figure(trials.load(args.exp_dir), args.swing, args.fps).write_html(out, auto_play=False)
    print(f"Wrote {out}")
    webbrowser.open(out.resolve().as_uri())
