"""Saved figure inputs and fixed-size previews for the hydrogen workflow."""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

# Sampled from the flat stage bars in the manuscript's SENSE overview figure.
# Use tints for distributions, not a second unrelated palette.
COLORS = {
    "indigo": "#3A4E8C",
    "burgundy": "#8C3A3A",
    "green": "#3A5F3A",
    "ochre": "#B38B2E",
    "ink": "#303030",
    "grey": "#737373",
    "grid": "#E7E7E7",
    "no_data": "#F0F0F0",
    "distribution": "#D8DCE8",
}


def _tint(hex_color: str, white: float) -> str:
    """Mix a master hue with white without introducing a second colour family."""
    channels = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(
        f"{round(channel * (1 - white) + 255 * white):02X}" for channel in channels
    )


# Category identities, not a sequential percentage scale. Interleave hue families;
# labels and hatching distinguish related shades in larger categorical legends.
CATEGORY_COLORS = [
    _tint(COLORS[hue], white)
    for white in (0.25, 0.5, 0.0, 0.65, 0.1)
    for hue in ("indigo", "burgundy", "green", "ochre")
] + [COLORS["grey"]]

# Keep shared country identities consistent in totals and contribution figures.
COUNTRY_COLORS = {
    "DE": COLORS["indigo"],
    "CN": COLORS["burgundy"],
    "US": COLORS["ochre"],
    "RU": _tint(COLORS["indigo"], 0.35),
    "CO": COLORS["green"],
    "CL": _tint(COLORS["burgundy"], 0.4),
    "CA": _tint(COLORS["ochre"], 0.4),
    "RoW": "#B9B9B9",
    "GLO": "#D4D4D4",
    "IAI Area, Russia and Eastern Europe": COLORS["indigo"],
}

TECHNOLOGY_COLORS = {
    "Biomass gasification": COLORS["green"],
    "Electrolysis": COLORS["indigo"],
    "SMR (biomethane)": COLORS["ochre"],
    "SMR (natural gas)": COLORS["burgundy"],
    "Coal gasification": COLORS["grey"],
}


def _analysis_signature() -> str:
    """Fingerprint local analysis code so old rankings cannot survive code changes."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def reuse_inputs(path: Path, *, context: dict, rebuild: bool = False, sources=None) -> bool:
    """Check a snapshot's manifest without loading tables or doing calculations.

    Return false if absent, explicitly rebuilt or different from supplied ``sources``
    metadata. Raise ``ValueError`` for schema/context mismatches. ``context`` is the
    expected project/version mapping; without ``sources``, source-data or analysis
    changes are not checked. This function does not start preparation itself.
    """
    path = Path(path)
    if rebuild or not path.is_file():
        return False
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema") != 1 or manifest["metadata"].get("context") != context:
            raise ValueError(
                "Saved inputs use another project/version or format. Set the rebuild flag explicitly."
            )
        if manifest.get("analysis_signature") != _analysis_signature():
            return False
        for name in manifest["tables"]:
            archive.getinfo(f"{name}.csv")
    source_directory = manifest["metadata"].get("source_directory")
    if sources is None and source_directory and Path(source_directory).is_dir():
        sources = [
            {"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(Path(source_directory).glob("*.csv"))
        ]
    if sources is not None and manifest["metadata"].get("sources") != sources:
        return False
    print(f"Reusing saved inputs: {path}. Skipping preparation.")
    return True


def save_inputs(path: Path, tables: dict[str, pd.DataFrame], metadata: dict) -> None:
    """Replace a snapshot atomically with CSV tables and a JSON metadata manifest.

    Table indexes are omitted. ``metadata`` must be JSON-serializable with finite
    numbers; the caller must exclude credentials and other secrets. The function
    does not sanitize metadata. Preserve an existing snapshot if serialization fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "analysis_signature": _analysis_signature(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "metadata": metadata,
        "tables": list(tables),
    }
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
            for name, table in tables.items():
                with archive.open(f"{name}.csv", "w") as raw:
                    with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
                        table.to_csv(stream, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved plotting inputs: {path}")


def load_inputs(path: Path, *, context: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return named DataFrames and metadata from a compatible figure snapshot.

    Require an exact project/version ``context`` match. Raise ``FileNotFoundError``
    if absent or ``ValueError`` for schema/context mismatches; never rebuild inputs.
    Preserve literal NA country codes and parse floating-point values for round trips.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No saved plotting inputs at {path}. Run the corresponding preparation stage once."
        )
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema") != 1:
            raise ValueError("Unsupported figure-input format; rerun preparation.")
        if manifest.get("analysis_signature") != _analysis_signature():
            raise ValueError("Saved inputs predate analysis code changes. Rerun preparation.")
        metadata = manifest["metadata"]
        if metadata.get("context") != context:
            raise ValueError("Saved figures belong to another project/version; rerun preparation.")
        tables = {}
        for name in manifest["tables"]:
            with archive.open(f"{name}.csv") as stream:
                # Preserve country codes such as NA and literal indicator labels.
                tables[name] = pd.read_csv(
                    stream, keep_default_na=False, na_values=[""], float_precision="round_trip"
                )
    print(f"Loaded snapshot from {manifest['created_utc']}: {path}")
    return tables, metadata


def figure_style() -> dict:
    """Return the shared manuscript palette and typography for ``plt.rc_context``.

    Use Helvetica-style Arial, then open-source sans-serif fallbacks.
    All plots use upright 7 pt regular text and 8 pt bold panel headings. No font
    files are downloaded or bundled; DejaVu Sans is available with Matplotlib.
    """
    from cycler import cycler
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(name for name in ("Arial", "Liberation Sans", "DejaVu Sans") if name in available)
    return {
        "font.family": family,
        "font.size": 7,
        "font.weight": "normal",
        "font.style": "normal",
        "font.stretch": "normal",
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.fallback": "stixsans",
        "mathtext.default": "regular",
        "axes.labelsize": 7,
        "axes.labelweight": "normal",
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "figure.titlesize": 8,
        "figure.titleweight": "bold",
        "figure.labelsize": 7,
        "figure.labelweight": "normal",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.65,
        "axes.edgecolor": "black",
        "axes.labelcolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.5,
        "axes.prop_cycle": cycler(color=[COLORS["indigo"], COLORS["burgundy"]]),
        "hatch.linewidth": 0.35,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": None,
    }


def export_figure(fig, stem: Path, *, preview_scale: float = 1.0, preview: bool = False) -> None:
    """Export an uncropped PDF/SVG canvas and display it at its CSS-mm dimensions.

    By default export without displaying. With preview=True, preview zoom affects only the notebook display. CSS millimetres are nominal:
    browser zoom and OS/display scaling determine the physical on-screen size.
    Overwrite ``stem.pdf`` and ``stem.svg``, then close the Figure. Reject nonfinite
    or nonpositive ``preview_scale`` with ``ValueError``.
    """
    import math

    import matplotlib.pyplot as plt

    if not math.isfinite(preview_scale) or preview_scale <= 0:
        raise ValueError("preview_scale must be finite and positive.")
    try:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "svg"):
            fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches=None, dpi=600)
        if not preview:
            return
        from IPython.display import HTML, display

        width, height = fig.get_size_inches() * 25.4
        buffer = io.StringIO()
        # Inline paths preserve Matplotlib's exact font metrics even when the browser
        # does not have the selected font installed. The exported SVG retains editable text.
        with plt.rc_context({"svg.fonttype": "path"}):
            fig.savefig(buffer, format="svg", bbox_inches=None)
        svg = buffer.getvalue()
        svg = svg[svg.index("<svg") :]
        # Explicit dimensions prevent notebook frontends from shrinking wide artwork.
        svg = svg.replace("<svg ", '<svg style="width:100%;height:100%;max-width:none" ', 1)
        display(
            HTML(
                f'<div style="overflow-x:auto"><div style="width:{width * preview_scale:.3f}mm;'
                f'height:{height * preview_scale:.3f}mm;flex:none">{svg}</div></div>'
                f"<p>Export: {width:.1f} × {height:.1f} mm · preview {preview_scale:g}×. "
                "At 1×, this ruler is nominally 50 mm; measure it to check your screen scaling.</p>"
                f'<div style="width:{50 * preview_scale}mm;border-top:1px solid #555;'
                'height:4px;border-left:1px solid #555;border-right:1px solid #555"></div>'
            )
        )
    finally:
        plt.close(fig)
