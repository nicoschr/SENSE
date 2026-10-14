"""US fatal-accident contributions, separately by activity and country."""

from __future__ import annotations

import json
import textwrap
from string import ascii_uppercase

import numpy as np
import pandas as pd

from .contribution_data import select_activity_groups
from .figure_io import CATEGORY_COLORS, COUNTRY_COLORS
from .study_protocol import load_protocol

OTHER_ACTIVITY = "Other activities"
PRODUCTION_COUNTRY = "US"
PALETTE = CATEGORY_COLORS
HATCHES = [
    "//",
    "..",
    "++",
    "xx",
    "--",
    "\\\\",
    "oo",
    "||",
    "/",
    "+",
    "x",
    "o",
    "-",
    "\\",
    ".",
    "//",
    "++",
    "..",
    "xx",
    "||",
    "--",
]
SEGMENT_EDGE = (0.15, 0.15, 0.15, 0.42)


# Display labels only; exact names, products and units are retained in the tables.
SHORT_NAMES = {
    "diesel production, low-sulfur, petroleum refinery operation": "Low-sulfur diesel production",
    "natural gas venting from petroleum/natural gas production": "Natural-gas venting (oil/gas production)",
    "natural gas production, liquefied": "Liquefied natural-gas production",
    "natural gas, high pressure, domestic supply with seasonal storage": "Natural-gas supply with seasonal storage",
    "wind turbine construction, 4.5MW, onshore": "Onshore wind-turbine construction (4.5 MW)",
    "treatment of sulfidic tailings, from copper mine operation, tailings impoundment": "Copper-mine sulfidic tailings treatment",
    "poplar chips, 50 wt% moisture": "Poplar chips (50% moisture)",
    "hard coal mine operation and hard coal preparation": "Hard-coal mining and preparation",
    "hydrogen production, steam methane reforming": "Hydrogen production: natural-gas SMR",
    "hydrogen production, coal gasification": "Hydrogen production: coal gasification",
    "transport, freight, train, diesel": "Rail freight, diesel",
    "transport, freight, train, electric": "Rail freight, electric",
    "chemical factory construction, organics": "Organic-chemical factory construction",
    "heat and power co-generation, natural gas, conventional power plant, 100MW electrical": "Natural-gas CHP (heat output)",
    "electricity production, hard coal": "Hard-coal electricity production",
    "anaerobic digestion of animal manure, with biogenic carbon uptake": "Manure digestion, with biogenic carbon uptake",
    "anaerobic digestion of agricultural residues, with biogenic carbon uptake": "Crop-residue digestion, with biogenic carbon uptake",
    "transport, pipeline, onshore, long distance, natural gas": "Natural-gas pipeline transport, onshore",
}


def display_label(activity):
    name, product, _ = json.loads(activity)
    if name == "petroleum and gas production, onshore":
        output = "natural gas" if product == "natural gas, high pressure" else product
        return f"Onshore petroleum/gas ({output})"
    label = SHORT_NAMES.get(name, name)
    return label[:1].upper() + label[1:]


def build_display_table(data):
    """Pool only for display; retain signed totals and stable activity/location keys."""
    import pycountry
    from matplotlib.colors import to_hex, to_rgb

    data = data[data.country.eq(PRODUCTION_COUNTRY)].copy()
    if data.empty:
        raise ValueError("No US hydrogen contributions found")
    grouped, retained = select_activity_groups(data)
    if len(retained) > len(PALETTE):
        raise ValueError("Too many activity groups for the contribution palette")
    # Number first appearances in panel order, independently of the colour map.
    panels = [panel["id"] for panel in load_protocol()["panels"]]
    panels += [panel for panel in data.panel.unique() if panel not in panels]
    activity_numbers: dict[str, int] = {}
    for panel in panels:
        present = set(grouped.loc[grouped.panel.eq(panel), "activity_group"])
        for activity in retained:
            if activity in present and activity not in activity_numbers:
                activity_numbers[activity] = len(activity_numbers) + 1
    grouped["magnitude"] = grouped.percent.abs()
    countries = {country.alpha_2 for country in pycountry.countries}
    # Collapse recognized subnational codes to their parent country; keep other
    # geographic aggregates (GLO, RoW, RER, etc.) exactly as labelled.
    grouped["plot_location"] = grouped.location.map(
        lambda location: (
            location.split("-", 1)[0] if location.split("-", 1)[0] in countries else location
        )
    )
    location_totals = grouped.groupby(["activity_group", "plot_location"]).magnitude.sum()
    legend = []
    display_frames = []
    keys = ["fu_index", "technology", "country", "scenario", "panel", "sign", "contribution_unit"]
    for index, activity in enumerate([*retained, OTHER_ACTIVITY]):
        subset = grouped[grouped.activity_group.eq(activity)].copy()
        if subset.empty:
            continue
        is_other = activity == OTHER_ACTIVITY
        locations = (
            location_totals.loc[activity].sort_values(ascending=False, kind="stable").index.tolist()
        )
        # Every country/region remains a separate segment, including within the
        # residual activity group. Never pool small countries into a remainder.
        subset["location_group"] = subset.plot_location
        base = "#DDDDDD" if is_other else PALETTE[index]
        shades = {}
        for rank, location in enumerate(locations):
            # Stable country shades across all panels/scenarios. A logarithmic
            # ramp keeps the leading countries distinguishable in long tails.
            tint = 0.78 * np.log1p(rank) / np.log(len(locations)) if len(locations) > 1 else 0
            rgb = np.array(to_rgb(base))
            shade = rgb * (1 - tint) + tint
            shades[location] = to_hex((float(shade[0]), float(shade[1]), float(shade[2])))
        summary = subset.groupby([*keys, "location_group"], as_index=False)[
            ["contribution", "percent"]
        ].sum()
        summary["activity_group"] = activity
        summary["activity_number"] = activity_numbers[activity] if not is_other else 0
        summary["color"] = summary.location_group.map(shades)
        display_frames.append(summary)
        legend.append(
            {
                "activity": activity,
                "number": activity_numbers[activity] if not is_other else 0,
                "color": base,
                "hatch": "" if is_other else HATCHES[index],
                "label": OTHER_ACTIVITY if is_other else display_label(activity),
                "shades": shades,
            }
        )
    legend.sort(key=lambda entry: entry["number"] or len(activity_numbers) + 1)
    table = pd.concat(display_frames, ignore_index=True)
    keys = ["fu_index", "panel", "sign"]
    for column in ("contribution", "percent"):
        original = data.groupby(keys)[column].sum().sort_index()
        displayed = table.groupby(keys)[column].sum().sort_index()
        if not original.index.equals(displayed.index) or not np.allclose(
            original, displayed, rtol=1e-12, atol=1e-10
        ):
            raise ValueError("Display grouping changed signed totals")
    return table, legend


def label_ink(color):
    """Choose black or white by relative-luminance contrast on an sRGB fill."""
    from matplotlib.colors import to_rgb

    rgb = np.array(to_rgb(color))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ [0.2126, 0.7152, 0.0722]
    return "black" if (luminance + 0.05) / 0.05 >= 1.05 / (luminance + 0.05) else "white"


def sum_activity_locations(table, legend):
    """Sum location detail for display while retaining positive and negative terms."""
    keys = [
        "fu_index",
        "technology",
        "country",
        "scenario",
        "panel",
        "sign",
        "contribution_unit",
        "activity_group",
        "activity_number",
    ]
    result = table.groupby(keys, as_index=False)[["contribution", "percent"]].sum()
    result["color"] = result.activity_group.map(
        {entry["activity"]: entry["color"] for entry in legend}
    )
    result["hatch"] = result.activity_group.map(
        {entry["activity"]: entry.get("hatch", "") for entry in legend}
    )
    result["activity_label"] = result.activity_number.map(
        lambda number: ascii_uppercase[number - 1] if number else "Other"
    )
    if result.color.isna().any():
        raise ValueError("Missing activity colour")
    return result


def positive_display_table(table):
    """Display positive contributions only, normalized to 100% in each bar."""
    result = table.loc[table.contribution.gt(0)].copy()
    totals = result.groupby(["fu_index", "panel"]).contribution.transform("sum")
    if result.empty or not np.isfinite(totals).all() or totals.le(0).any():
        raise ValueError("Contribution display requires positive totals")
    result["percent"] = 100 * result.contribution / totals
    return result


def sum_country_activities(table, *, label_threshold=5):
    """Retain locations reaching 5% in any bar and pool the rest for display."""
    import pycountry

    keys = [
        "fu_index",
        "technology",
        "country",
        "scenario",
        "panel",
        "sign",
        "contribution_unit",
        "location_group",
    ]
    result = table.groupby(keys, as_index=False)[["contribution", "percent"]].sum()
    magnitude = result.assign(magnitude=result.percent.abs())
    per_bar = magnitude.groupby(["fu_index", "panel", "location_group"]).magnitude.sum()
    prominent = set(
        per_bar.groupby("location_group").max().loc[lambda x: x.ge(label_threshold)].index
    )
    order = (
        magnitude.groupby("location_group")
        .magnitude.sum()
        .sort_values(ascending=False, kind="stable")
        .index.tolist()
    )
    order = [loc for loc in order if loc in prominent]
    other = "Other countries/regions"
    has_other = (~result.location_group.isin(prominent)).any()
    result["location_group"] = result.location_group.where(
        result.location_group.isin(prominent), other
    )
    result = result.groupby(keys, as_index=False)[["contribution", "percent"]].sum()
    if has_other:
        order.append(other)
    if len(prominent) > len(PALETTE):
        raise ValueError("Too many labelled countries for the contribution palette")
    iai = "IAI Area, Russia and Eastern Europe"
    names = {
        "RoW": "Rest of world",
        "GLO": "Global",
        "RER": "Europe",
        iai: "Russia and Eastern Europe",
    }
    colors, labels, hatches, legend = {}, {}, {}, []
    for rank, location in enumerate(order):
        if location == other:
            color, hatch, inline, label = "#DDDDDD", "", "Other", other
        else:
            color = COUNTRY_COLORS.get(location, PALETTE[rank])
            country = pycountry.countries.get(alpha_2=location)
            name = names.get(location, country.name if country else location)
            inline = "IAI" if location == iai else location
            label = inline if name == inline else f"{inline}  {name}"
            hatch = HATCHES[rank]
        colors[location], labels[location], hatches[location] = color, inline, hatch
        legend.append({"label": label, "color": color, "hatch": hatch})
    result["color"] = result.location_group.map(colors)
    result["inline_label"] = result.location_group.map(labels)
    result["hatch"] = result.location_group.map(hatches)
    return result, legend, order


def plot_contributions(table, legend, spec, technologies):
    """Two panels for US fatal-accident scores, with five paired routes each."""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle

    if set(table.panel) != {"fatal_accidents"} or set(table.country) != {PRODUCTION_COUNTRY}:
        raise ValueError("Contribution figure requires only fatal accidents in the United States")
    table = positive_display_table(table)
    activities = sum_activity_locations(table, legend)
    activities["inline_label"] = activities.activity_label
    countries, country_legend, country_order = sum_country_activities(table)
    activity_legend = [
        {
            "label": (f"{ascii_uppercase[entry['number'] - 1]}  " if entry["number"] else "")
            + entry["label"],
            "color": entry["color"],
            "hatch": entry.get("hatch", ""),
        }
        for entry in legend
    ]
    panels = [
        (
            "Fatal-accident risk scores (US): by activity",
            activities,
            "activity_group",
            [entry["activity"] for entry in legend],
            activity_legend,
            "Activities",
        ),
        (
            "Fatal-accident risk scores (US): by country",
            countries,
            "location_group",
            country_order,
            country_legend,
            "Countries/regions",
        ),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(180 / 25.4, 180 / 25.4))
    fig.subplots_adjust(left=0.175, right=0.665, top=0.96, bottom=0.06, hspace=0.23)
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    renderer = FigureCanvasAgg(fig).get_renderer()
    label_font = FontProperties(family=plt.rcParams["font.family"], size=7)
    for letter, ax, (title, panel_data, group, order, entries, heading) in zip(
        ("(a)", "(b)"), axes, panels, strict=True
    ):
        ax.set_xlim(0, 100)
        ax.set_ylim(len(technologies) - 0.5, -0.5)
        for ti, technology in enumerate(technologies):
            for si, scenario in enumerate(("baseline", "regionalized")):
                bar_y = ti + (si - 0.5) * 0.38
                bar = panel_data[
                    panel_data.technology.eq(technology) & panel_data.scenario.eq(scenario)
                ]
                if bar.empty:
                    raise ValueError(f"Missing contribution bar: {technology}, {scenario}")
                parts_by_key = {
                    (getattr(part, group), part.sign): part for part in bar.itertuples()
                }
                for sign in ("positive",):
                    left = 0.0
                    for category in order:
                        part = parts_by_key.get((category, sign))
                        if part is None:
                            continue
                        ax.barh(
                            bar_y,
                            part.percent,
                            left=left,
                            height=0.32,
                            color=part.color,
                            hatch=part.hatch,
                            edgecolor=SEGMENT_EDGE,
                            # Keep thin segment borders from obscuring tiny countries.
                            linewidth=min(
                                0.3, abs(part.percent) * ax.bbox.width / fig.dpi * 72 / 100 * 0.4
                            ),
                            zorder=2,
                        )
                        label = part.inline_label
                        text_width, _, _ = renderer.get_text_width_height_descent(
                            label, label_font, ismath=False
                        )
                        minimum_width = (text_width + 4 * fig.dpi / 72) / ax.bbox.width * 100
                        if label and abs(part.percent) >= max(5, minimum_width):
                            ax.text(
                                left + part.percent / 2,
                                bar_y,
                                label,
                                color=label_ink(part.color),
                                ha="center",
                                va="center",
                                fontsize=7,
                                bbox={"facecolor": part.color, "edgecolor": "none", "pad": 0.3},
                                zorder=3,
                            )
                        left += part.percent
                ax.text(
                    1.012,
                    bar_y,
                    "O" if si == 0 else "R",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    fontsize=7,
                )
            if ti < len(technologies) - 1:
                ax.axhline(ti + 0.5, color="#A6A6A6", linewidth=0.5, linestyle=(0, (3, 3)))
        ax.set_yticks(
            range(len(technologies)),
            [textwrap.fill(spec["technology_labels"].get(t, t), 18) for t in technologies],
            fontsize=7,
        )
        ax.tick_params(axis="y", length=0, pad=5)
        ax.text(
            0,
            1.035,
            f"{letter}  {title}",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="baseline",
        )
        ax.set_xlabel("Contribution (%)", fontsize=7)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", labelsize=7)
        ax.axvline(0, color="black", linewidth=0.5, zorder=4)
        ax.grid(axis="x", color="#EAEAEA", linewidth=0.35)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.65)
        box = ax.get_position()
        key = fig.add_axes((0.715, box.y0, 0.275, box.height))
        key.set_axis_off()
        # Identical axes heights and baseline coordinates align panel/legend headings.
        key.text(0, 1.035, heading, weight="bold", fontsize=8, va="baseline")
        labels = [textwrap.fill(entry["label"], 30) for entry in entries]
        height_points = key.bbox.height / fig.dpi * 72
        y = 0.985
        for entry, label in zip(entries, labels, strict=True):
            text = key.text(0.10, y, label, va="top", fontsize=7, linespacing=1.15)
            key.add_patch(
                Rectangle(
                    (0, y - 6.0 / height_points),
                    0.07,
                    5.5 / height_points,
                    facecolor=entry["color"],
                    hatch=entry["hatch"],
                    edgecolor=SEGMENT_EDGE,
                    linewidth=0.3,
                )
            )
            # Use the rendered text height and a fixed 3 pt gap in both legends.
            # Do not stretch a short legend to fill its panel's height.
            y -= text.get_window_extent(renderer).height / key.bbox.height + 3 / height_points
        if y < -0.02:
            raise ValueError("Contribution legend exceeds panel height at the fixed font size")
    return fig
