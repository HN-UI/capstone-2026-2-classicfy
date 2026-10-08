"""Render figures afresh and check displayed text stays inside the canvas.

This complements visual review; it does not detect all text overlap or assess
whether a chart is scientifically meaningful.
"""

import argparse
import json
from pathlib import Path

import plot_extended_sequences as plots
from matplotlib.backends.backend_agg import FigureCanvasAgg


def check(out):
    original_finish = plots.finish
    reviewed, violations = [], []

    def inspected_finish(fig, path, *args, **kwargs):
        original_finish(fig, path, *args, **kwargs)
        # Closing the pyplot manager does not destroy the retained Figure.
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        texts = list(fig.texts)
        for ax in fig.axes:
            texts.extend([ax.title, ax.xaxis.label, ax.yaxis.label,
                          ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()])
            texts.extend(ax.texts)
            if ax.get_legend() is not None:
                texts.extend(ax.get_legend().get_texts())
            for axis in (ax.xaxis, ax.yaxis):
                lower, upper = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks():
                    if lower <= tick.get_loc() <= upper:
                        texts.extend((tick.label1, tick.label2))
        for text in texts:
            if not text.get_visible() or not text.get_text().strip():
                continue
            box = text.get_window_extent(renderer)
            if box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.width+2 or box.y1 > fig.bbox.height+2:
                violations.append({"figure": str(Path(path).relative_to(out)), "text": text.get_text(),
                                   "bounds": list(box.extents), "canvas": [fig.bbox.width, fig.bbox.height]})
        reviewed.append(str(Path(path).relative_to(out)))

    plots.finish = inspected_finish
    try:
        plots.render(out)
    finally:
        plots.finish = original_finish
    result = {"status": "passed" if not violations else "failed", "figures_checked": len(reviewed),
              "plot_source_sha256": plots.sha(Path(plots.__file__)),
              "figure_manifest_sha256": plots.sha(out / "figure_manifest.json"),
              "check": "Visible titles, captions, axis labels, tick labels, legends and annotations fit inside canvas",
              "limitation": "Does not detect every text overlap; visual review is also required",
              "violations": violations, "figures": reviewed}
    (out / "layout_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if violations:
        raise SystemExit("Figure text is clipped; see layout_audit.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "analysis/sequence_followup")
    check(parser.parse_args().out)
