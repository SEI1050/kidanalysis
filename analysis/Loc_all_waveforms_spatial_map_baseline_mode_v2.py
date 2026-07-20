from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# SETTINGS
# ============================================================

DATA_ROOT = Path("/Volumes/NO NAME/data/20260709")

OUTPUT_DIR = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260709/"
    "all_waveforms_spatial_map"
)

TARGET_DIR_PATTERN = "5.501GHz_*"
WAVEFORM_GLOB = "wf_*.npz"


# ------------------------------------------------------------
# Baseline processing mode
# ------------------------------------------------------------
# "none":
#     No baseline subtraction. Raw waveforms are plotted.
#
# "subtract":
#     For each event, calculate the mean of the first
#     BASELINE_FRACTION of the samples and subtract it
#     from the entire waveform.
#
BASELINE_MODE = "subtract"
BASELINE_FRACTION = 0.10

# In baseline-subtracted mode, draw a y=0 guide line.
SHOW_ZERO_LINE = True


# ------------------------------------------------------------
# Spatial-map appearance
# ------------------------------------------------------------

SPATIAL_FIGSIZE = (26, 20)
SPATIAL_DPI = 220

SPATIAL_CELL_WIDTH_FRACTION = 0.92
SPATIAL_CELL_HEIGHT_FRACTION = 0.78

SPATIAL_INNER_GAP_FRACTION = 0.05
SPATIAL_MAX_COLUMNS_PER_POSITION = 3

SPATIAL_Z_INCREASES_UPWARD = True
SPATIAL_EQUAL_ASPECT = True


# ------------------------------------------------------------
# Waveform drawing
# ------------------------------------------------------------

# None: plot every event.
# Integer: plot that many events, chosen at nearly equal intervals.
MAX_EVENTS_TO_PLOT = None

EVENT_LINEWIDTH = 0.35
EVENT_ALPHA = 0.08
MEAN_LINEWIDTH = 2.0

# "per_folder":
#     Each panel has its own y-axis range.
#
# "per_position":
#     Panels measured at the same (x, z) share a y-axis range.
#
# "global":
#     Every panel shares the same y-axis range.
#
WAVEFORM_AXIS_MODE = "per_position"

# "full":
#     Set the y-axis from the minimum and maximum of every event
#     that is actually plotted. No plotted waveform is cut off.
#
# "robust":
#     Use percentiles to suppress rare outliers. Some extreme
#     events can be outside the visible range.
#
YLIM_MODE = "full"

YLIM_PERCENTILES = (0.5, 99.5)
YLIM_PADDING_FRACTION = 0.15
YLIM_SAMPLE_STRIDE = 10

# Small horizontal margin so the first and last samples do not
# touch the panel borders.
TIME_PADDING_FRACTION = 0.01

# Panel annotations. The physical x/z position is written once
# above each position cell, instead of being repeated above every
# mini-panel.
SHOW_POSITION_LABEL = True
SHOW_PANEL_TAG = True
SHOW_EVENT_COUNT = True
SHOW_CHANNEL_LABEL = False

POSITION_LABEL_FONTSIZE = 8
PANEL_TAG_FONTSIZE = 6
PANEL_TICK_FONTSIZE = 5


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class WaveformResult:
    folder_name: str
    waveform_path: Path
    x_mm: float
    z_mm: float
    tag: str
    time_ns: np.ndarray
    ch0: np.ndarray
    ch1: np.ndarray
    mean_ch0: np.ndarray
    mean_ch1: np.ndarray
    sample_rate: float
    baseline_samples: int
    base_ch0: np.ndarray
    base_ch1: np.ndarray


# ============================================================
# FOLDER NAME PARSING
# ============================================================

_FOLDER_RE = re.compile(
    r"^(?P<freq>[\d.]+)GHz_"
    r"z=(?P<z>[\d.]+)mm_"
    r"x=(?P<x>[\d.]+)mm"
    r"(?:_(?P<tag>.+))?$"
)


def parse_folder_info(folder_name: str) -> tuple[float, float, str]:
    match = _FOLDER_RE.match(folder_name)

    if match is None:
        raise ValueError(
            f"Could not parse folder name: {folder_name}"
        )

    z_mm = float(match.group("z"))
    x_mm = float(match.group("x"))
    tag = match.group("tag") or ""

    return x_mm, z_mm, tag


def friendly_title(folder_name: str) -> str:
    x_mm, z_mm, tag = parse_folder_info(folder_name)

    if tag:
        return (
            f"x={x_mm:g} mm, z={z_mm:g} mm\n"
            f"({tag})"
        )

    return f"x={x_mm:g} mm, z={z_mm:g} mm"


# ============================================================
# FILE DISCOVERY
# ============================================================

def discover_waveform_files(
    data_root: Path,
) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []

    for folder in sorted(data_root.glob(TARGET_DIR_PATTERN)):
        if not folder.is_dir():
            continue

        waveform_files = sorted(folder.glob(WAVEFORM_GLOB))

        if not waveform_files:
            print(f"[skip] no waveform file in {folder.name}")
            continue

        if len(waveform_files) > 1:
            print(
                f"[info] {folder.name}: multiple waveform files found; "
                f"use first -> {waveform_files[0].name}"
            )

        pairs.append(
            (folder.name, waveform_files[0])
        )

    pairs.sort(
        key=lambda pair: (
            parse_folder_info(pair[0])[1],  # z
            parse_folder_info(pair[0])[0],  # x
            parse_folder_info(pair[0])[2],  # tag
        )
    )

    return pairs


# ============================================================
# WAVEFORM LOADING
# ============================================================

def load_waveform(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as data:
        print(f"[load] {path}")
        print("       npz keys:", list(data.keys()))

        ch0 = np.asarray(data["ch0"], dtype=float)
        ch1 = np.asarray(data["ch1"], dtype=float)

        if "sample_rate" in data:
            sample_rate = float(
                np.asarray(data["sample_rate"]).squeeze()
            )
        else:
            sample_rate = 2.5e9

    if ch0.shape != ch1.shape:
        raise ValueError(
            f"ch0 shape {ch0.shape} and "
            f"ch1 shape {ch1.shape} differ"
        )

    if ch0.ndim == 1:
        ch0 = ch0[np.newaxis, :]
        ch1 = ch1[np.newaxis, :]

    if ch0.ndim != 2:
        raise ValueError(
            f"Waveform array must be 2D, got {ch0.shape}"
        )

    # Convert to (events, samples), if needed.
    if ch0.shape[0] > ch0.shape[1]:
        ch0 = ch0.T
        ch1 = ch1.T

    return ch0, ch1, sample_rate


# ============================================================
# BASELINE PROCESSING
# ============================================================

def apply_baseline_mode(
    ch0: np.ndarray,
    ch1: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """
    Apply the selected baseline mode.

    Returns
    -------
    processed_ch0, processed_ch1:
        Waveforms after the selected processing.

    base_ch0, base_ch1:
        One baseline value per event. Shape: (n_events,)

    baseline_samples:
        Number of samples used to estimate the baseline.
    """

    valid_modes = {
        "none",
        "subtract",
    }

    if BASELINE_MODE not in valid_modes:
        raise ValueError(
            f"Unknown BASELINE_MODE={BASELINE_MODE!r}. "
            f"Choose from {sorted(valid_modes)}."
        )

    n_samples = ch0.shape[1]

    baseline_samples = max(
        1,
        int(round(n_samples * BASELINE_FRACTION)),
    )

    # One baseline value per event.
    base_ch0 = np.mean(
        ch0[:, :baseline_samples],
        axis=1,
    )
    base_ch1 = np.mean(
        ch1[:, :baseline_samples],
        axis=1,
    )

    if BASELINE_MODE == "none":
        processed_ch0 = ch0.copy()
        processed_ch1 = ch1.copy()

    elif BASELINE_MODE == "subtract":
        processed_ch0 = (
            ch0 - base_ch0[:, np.newaxis]
        )
        processed_ch1 = (
            ch1 - base_ch1[:, np.newaxis]
        )

    return (
        processed_ch0,
        processed_ch1,
        base_ch0,
        base_ch1,
        baseline_samples,
    )


def baseline_mode_label() -> str:
    if BASELINE_MODE == "none":
        return "raw waveform; no baseline subtraction"

    if BASELINE_MODE == "subtract":
        return (
            "event-by-event baseline subtraction "
            f"using first {BASELINE_FRACTION:.0%} mean"
        )

    return BASELINE_MODE


def output_mode_name() -> str:
    if BASELINE_MODE == "none":
        return "raw"

    if BASELINE_MODE == "subtract":
        percentage = int(
            round(BASELINE_FRACTION * 100)
        )
        return (
            f"baseline_subtracted_"
            f"first{percentage}pct"
        )

    return BASELINE_MODE


# ============================================================
# RESULT CONSTRUCTION
# ============================================================

def build_result(
    folder_name: str,
    waveform_path: Path,
) -> WaveformResult:
    x_mm, z_mm, tag = parse_folder_info(
        folder_name
    )

    raw_ch0, raw_ch1, sample_rate = load_waveform(
        waveform_path
    )

    n_events, n_samples = raw_ch0.shape

    time_ns = (
        np.arange(n_samples, dtype=float)
        / sample_rate
        * 1e9
    )

    (
        ch0,
        ch1,
        base_ch0,
        base_ch1,
        baseline_samples,
    ) = apply_baseline_mode(
        raw_ch0,
        raw_ch1,
    )

    # The mean waveform is calculated after the selected processing.
    mean_ch0 = np.mean(ch0, axis=0)
    mean_ch1 = np.mean(ch1, axis=0)

    baseline_duration_ns = (
        baseline_samples
        / sample_rate
        * 1e9
    )

    print(
        f"       events={n_events}, "
        f"samples={n_samples}, "
        f"sample_rate={sample_rate:g} Hz"
    )
    print(
        f"       baseline mode={BASELINE_MODE}, "
        f"samples={baseline_samples}, "
        f"duration={baseline_duration_ns:.3f} ns"
    )

    return WaveformResult(
        folder_name=folder_name,
        waveform_path=waveform_path,
        x_mm=x_mm,
        z_mm=z_mm,
        tag=tag,
        time_ns=time_ns,
        ch0=ch0,
        ch1=ch1,
        mean_ch0=mean_ch0,
        mean_ch1=mean_ch1,
        sample_rate=sample_rate,
        baseline_samples=baseline_samples,
        base_ch0=base_ch0,
        base_ch1=base_ch1,
    )


# ============================================================
# AXIS HELPERS
# ============================================================

def choose_event_indices(
    n_events: int,
    max_events: int | None,
) -> np.ndarray:
    if (
        max_events is None
        or max_events >= n_events
    ):
        return np.arange(n_events)

    return np.unique(
        np.linspace(
            0,
            n_events - 1,
            max_events,
            dtype=int,
        )
    )


def minimum_positive_gap(
    values: list[float],
    fallback: float,
) -> float:
    unique_values = np.array(
        sorted(set(values)),
        dtype=float,
    )

    if unique_values.size < 2:
        return fallback

    differences = np.diff(unique_values)
    differences = differences[
        differences > 0
    ]

    if differences.size == 0:
        return fallback

    return float(np.min(differences))


def finite_minmax(
    values: np.ndarray,
) -> tuple[float, float] | None:
    finite = np.isfinite(values)

    if not np.any(finite):
        return None

    minimum = float(
        np.min(
            values,
            where=finite,
            initial=np.inf,
        )
    )
    maximum = float(
        np.max(
            values,
            where=finite,
            initial=-np.inf,
        )
    )

    return minimum, maximum


def robust_waveform_ylim(
    results: list[WaveformResult],
    channel: str,
) -> tuple[float, float]:
    """
    Determine the y-axis limits.

    YLIM_MODE == "full":
        Use the full range of every event that is actually plotted.
        This prevents visible waveforms from being cut off.

    YLIM_MODE == "robust":
        Use percentile limits from a sampled event cloud, while
        always including the complete mean waveform.
    """

    valid_modes = {
        "full",
        "robust",
    }

    if YLIM_MODE not in valid_modes:
        raise ValueError(
            f"Unknown YLIM_MODE={YLIM_MODE!r}. "
            f"Choose from {sorted(valid_modes)}."
        )

    lower = np.inf
    upper = -np.inf

    if YLIM_MODE == "full":
        for result in results:
            if channel == "ch0":
                waveforms = result.ch0
                mean_trace = result.mean_ch0

            elif channel == "ch1":
                waveforms = result.ch1
                mean_trace = result.mean_ch1

            else:
                raise ValueError(
                    f"Unknown channel: {channel}"
                )

            event_indices = choose_event_indices(
                waveforms.shape[0],
                MAX_EVENTS_TO_PLOT,
            )

            if (
                MAX_EVENTS_TO_PLOT is None
                or len(event_indices) == waveforms.shape[0]
            ):
                plotted_waveforms = waveforms
            else:
                plotted_waveforms = waveforms[event_indices]

            event_range = finite_minmax(
                plotted_waveforms
            )

            if event_range is not None:
                event_minimum, event_maximum = event_range
                lower = min(lower, event_minimum)
                upper = max(upper, event_maximum)

            mean_range = finite_minmax(
                mean_trace
            )

            if mean_range is not None:
                mean_minimum, mean_maximum = mean_range
                lower = min(lower, mean_minimum)
                upper = max(upper, mean_maximum)

    elif YLIM_MODE == "robust":
        collected: list[np.ndarray] = []
        all_means: list[np.ndarray] = []

        for result in results:
            if channel == "ch0":
                waveforms = result.ch0
                mean_trace = result.mean_ch0

            elif channel == "ch1":
                waveforms = result.ch1
                mean_trace = result.mean_ch1

            else:
                raise ValueError(
                    f"Unknown channel: {channel}"
                )

            event_indices = choose_event_indices(
                waveforms.shape[0],
                min(waveforms.shape[0], 80),
            )

            sample_indices = np.arange(
                0,
                waveforms.shape[1],
                max(1, YLIM_SAMPLE_STRIDE),
            )

            sampled = waveforms[
                np.ix_(
                    event_indices,
                    sample_indices,
                )
            ].ravel()

            sampled = sampled[
                np.isfinite(sampled)
            ]

            if sampled.size > 0:
                collected.append(sampled)

            finite_mean = mean_trace[
                np.isfinite(mean_trace)
            ]

            if finite_mean.size > 0:
                collected.append(finite_mean)
                all_means.append(finite_mean)

        if collected:
            values = np.concatenate(collected)

            lower, upper = np.nanpercentile(
                values,
                YLIM_PERCENTILES,
            )

        if all_means:
            mean_values = np.concatenate(all_means)
            lower = min(
                float(lower),
                float(np.min(mean_values)),
            )
            upper = max(
                float(upper),
                float(np.max(mean_values)),
            )

    if not np.isfinite(lower) or not np.isfinite(upper):
        return (-1.0, 1.0)

    if BASELINE_MODE == "subtract":
        lower = min(float(lower), 0.0)
        upper = max(float(upper), 0.0)

    span = max(
        float(upper - lower),
        1e-12,
    )

    padding = (
        YLIM_PADDING_FRACTION * span
    )

    return (
        float(lower - padding),
        float(upper + padding),
    )


# ============================================================
# PANEL DRAWING
# ============================================================

def plot_waveform_panel(
    ax: plt.Axes,
    result: WaveformResult,
    channel: str,
    ylim: tuple[float, float],
) -> None:
    if channel == "ch0":
        waveforms = result.ch0
        mean_waveform = result.mean_ch0

    elif channel == "ch1":
        waveforms = result.ch1
        mean_waveform = result.mean_ch1

    else:
        raise ValueError(
            f"Unknown channel: {channel}"
        )

    event_indices = choose_event_indices(
        waveforms.shape[0],
        MAX_EVENTS_TO_PLOT,
    )

    for event_index in event_indices:
        ax.plot(
            result.time_ns,
            waveforms[event_index],
            color="gray",
            linewidth=EVENT_LINEWIDTH,
            alpha=EVENT_ALPHA,
            zorder=1,
        )

    ax.plot(
        result.time_ns,
        mean_waveform,
        color="black",
        linewidth=MEAN_LINEWIDTH,
        zorder=3,
    )

    if (
        SHOW_ZERO_LINE
        and BASELINE_MODE == "subtract"
    ):
        ax.axhline(
            0.0,
            color="tab:red",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            zorder=2,
        )

    # Do not repeat x/z as a title above every mini-panel.
    # The position is displayed once above the complete position cell.
    ax.set_title("")

    time_span = max(
        float(result.time_ns[-1] - result.time_ns[0]),
        1e-12,
    )
    time_padding = (
        TIME_PADDING_FRACTION * time_span
    )

    ax.set_xlim(
        result.time_ns[0] - time_padding,
        result.time_ns[-1] + time_padding,
    )
    ax.set_ylim(ylim)

    ax.grid(alpha=0.25)
    ax.tick_params(
        labelsize=PANEL_TICK_FONTSIZE,
        pad=1,
    )

    if SHOW_PANEL_TAG and result.tag:
        tag_label = result.tag.replace("_", " ")

        ax.text(
            0.02,
            0.97,
            tag_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=PANEL_TAG_FONTSIZE,
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.10",
                "fc": "white",
                "ec": "none",
                "alpha": 0.72,
            },
            zorder=5,
        )

    if SHOW_EVENT_COUNT:
        ax.text(
            0.98,
            0.97,
            f"N={waveforms.shape[0]}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=PANEL_TAG_FONTSIZE,
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.10",
                "fc": "white",
                "ec": "none",
                "alpha": 0.72,
            },
            zorder=5,
        )

    if SHOW_CHANNEL_LABEL:
        ax.text(
            0.02,
            0.04,
            channel,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_TAG_FONTSIZE,
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "0.7",
                "alpha": 0.8,
            },
            zorder=5,
        )


# ============================================================
# SPATIAL MAP
# ============================================================

def create_spatial_waveform_figure(
    results: list[WaveformResult],
    channel: str,
) -> plt.Figure:
    if not results:
        raise RuntimeError(
            "No waveform results to plot"
        )

    groups: dict[
        tuple[float, float],
        list[WaveformResult],
    ] = {}

    for result in results:
        groups.setdefault(
            (result.x_mm, result.z_mm),
            [],
        ).append(result)

    x_values = sorted(
        {result.x_mm for result in results}
    )
    z_values = sorted(
        {result.z_mm for result in results}
    )

    dx = minimum_positive_gap(
        x_values,
        fallback=1.0,
    )
    dz = minimum_positive_gap(
        z_values,
        fallback=1.0,
    )

    cell_width_data = (
        SPATIAL_CELL_WIDTH_FRACTION * dx
    )
    cell_height_data = (
        SPATIAL_CELL_HEIGHT_FRACTION * dz
    )

    x_margin = 0.6 * dx
    z_margin = 0.6 * dz

    figure = plt.figure(
        figsize=SPATIAL_FIGSIZE,
    )

    background_ax = figure.add_subplot(111)

    background_ax.set_xlim(
        min(x_values) - x_margin,
        max(x_values) + x_margin,
    )
    background_ax.set_ylim(
        min(z_values) - z_margin,
        max(z_values) + z_margin,
    )

    if not SPATIAL_Z_INCREASES_UPWARD:
        background_ax.invert_yaxis()

    if SPATIAL_EQUAL_ASPECT:
        background_ax.set_aspect(
            "equal",
            adjustable="box",
        )

    background_ax.grid(
        alpha=0.25,
        linestyle="--",
    )
    background_ax.set_xlabel(
        "Physical x position [mm]",
        fontsize=13,
    )
    background_ax.set_ylabel(
        "Physical z position [mm]",
        fontsize=13,
    )

    background_ax.set_title(
        "All waveform events arranged by "
        f"measurement position ({channel})",
        fontsize=18,
        weight="bold",
        pad=18,
    )

    figure.text(
        0.5,
        0.965,
        "Thin gray = waveform events   |   "
        "thick black = mean waveform   |   "
        f"baseline mode = {baseline_mode_label()}   |   "
        f"y-limit mode = {YLIM_MODE}",
        ha="center",
        va="top",
        fontsize=11,
    )

    global_ylim = robust_waveform_ylim(
        results,
        channel,
    )

    for (x_mm, z_mm), group in groups.items():
        group = sorted(
            group,
            key=lambda result: result.folder_name,
        )

        number_of_panels = len(group)

        number_of_columns = min(
            SPATIAL_MAX_COLUMNS_PER_POSITION,
            number_of_panels,
        )

        number_of_rows = math.ceil(
            number_of_panels
            / number_of_columns
        )

        gap_x = (
            SPATIAL_INNER_GAP_FRACTION
            * cell_width_data
        )
        gap_y = (
            SPATIAL_INNER_GAP_FRACTION
            * cell_height_data
        )

        panel_width = (
            cell_width_data
            - gap_x * (number_of_columns - 1)
        ) / number_of_columns

        panel_height = (
            cell_height_data
            - gap_y * (number_of_rows - 1)
        ) / number_of_rows

        cell_left = (
            x_mm - cell_width_data / 2
        )
        cell_bottom = (
            z_mm - cell_height_data / 2
        )

        guide_rectangle = plt.Rectangle(
            (cell_left, cell_bottom),
            cell_width_data,
            cell_height_data,
            fill=False,
            edgecolor="0.8",
            linewidth=0.8,
            linestyle=":",
            zorder=1,
        )

        background_ax.add_patch(
            guide_rectangle
        )

        if SHOW_POSITION_LABEL:
            label_y = (
                cell_bottom
                + cell_height_data
                + 0.025 * dz
            )

            background_ax.text(
                x_mm,
                label_y,
                f"x={x_mm:g} mm, z={z_mm:g} mm",
                ha="center",
                va="bottom",
                fontsize=POSITION_LABEL_FONTSIZE,
                weight="bold",
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.82,
                },
                zorder=10,
            )

        if WAVEFORM_AXIS_MODE == "per_position":
            position_ylim = robust_waveform_ylim(
                group,
                channel,
            )
        else:
            position_ylim = None

        for index, result in enumerate(group):
            row = index // number_of_columns
            column = index % number_of_columns

            left = (
                cell_left
                + column * (
                    panel_width + gap_x
                )
            )

            bottom = (
                cell_bottom
                + (
                    number_of_rows
                    - 1
                    - row
                )
                * (
                    panel_height + gap_y
                )
            )

            inset_ax = background_ax.inset_axes(
                [
                    left,
                    bottom,
                    panel_width,
                    panel_height,
                ],
                transform=background_ax.transData,
            )

            if WAVEFORM_AXIS_MODE == "global":
                ylim = global_ylim

            elif WAVEFORM_AXIS_MODE == "per_position":
                assert position_ylim is not None
                ylim = position_ylim

            elif WAVEFORM_AXIS_MODE == "per_folder":
                ylim = robust_waveform_ylim(
                    [result],
                    channel,
                )

            else:
                raise ValueError(
                    f"Unknown WAVEFORM_AXIS_MODE="
                    f"{WAVEFORM_AXIS_MODE!r}"
                )

            plot_waveform_panel(
                inset_ax,
                result,
                channel,
                ylim,
            )

    return figure


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    waveform_pairs = discover_waveform_files(
        DATA_ROOT
    )

    if not waveform_pairs:
        raise RuntimeError(
            "No waveform folders found."
        )

    results: list[WaveformResult] = []

    for folder_name, waveform_path in waveform_pairs:
        results.append(
            build_result(
                folder_name,
                waveform_path,
            )
        )

    mode_name = output_mode_name()

    pdf_path = (
        OUTPUT_DIR
        / f"all_waveforms_spatial_map_"
          f"{mode_name}.pdf"
    )

    with PdfPages(pdf_path) as pdf:
        for channel in ("ch0", "ch1"):
            figure = create_spatial_waveform_figure(
                results,
                channel,
            )

            png_path = (
                OUTPUT_DIR
                / f"all_waveforms_spatial_map_"
                  f"{channel}_{mode_name}.png"
            )

            figure.savefig(
                png_path,
                dpi=SPATIAL_DPI,
                bbox_inches="tight",
            )

            pdf.savefig(
                figure,
                bbox_inches="tight",
            )

            plt.close(figure)

            print("[saved]", png_path)

    print("[saved]", pdf_path)
    print("Done.")


if __name__ == "__main__":
    main()
