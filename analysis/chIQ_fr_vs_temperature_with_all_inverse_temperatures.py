from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, curve_fit


# =============================================================================
# DATA
# =============================================================================

# T [K], fr [Hz], standard error of fr [Hz]
temperature_k = np.array([7.41, 6.68, 6.10, 5.56, 4.74, 3.50])
fr_hz = np.array([
    5444033000.0,
    5464446000.0,
    5473735000.0,
    5485420000.0,
    5485600000.0,
    5489634000.0,
])
fr_se_hz = np.array([
    188348.6,
    139376.4,
    124655.4,
    100795.8,
    103458.6,
    94674.31,
])


# =============================================================================
# SETTINGS
# =============================================================================

# False: 通常の最小二乗法
# True : fr の標準誤差を重みとした重み付き最小二乗法
MODEL_FIT_USE_WEIGHTS = False

# -----------------------------------------------------------------------------
# 指定周波数と、その 1 sigma 不確かさ。
#
# TARGET_FR_GHZ  : 逆算したい共振周波数 [GHz]
# TARGET_FR_SE_HZ: 上の各周波数そのものの測定誤差 [Hz]
#                  読み出し周波数を設定値として扱い、誤差を無視する場合は 0.0。
#
# 例:
# TARGET_FR_GHZ = [5.451, 5.476, 5.490, 5.501]
# TARGET_FR_SE_HZ = [0.0, 0.0, 0.0, 0.0]
# -----------------------------------------------------------------------------
TARGET_FR_GHZ = [5.476]
TARGET_FR_SE_HZ = [0.0]

# 外挿を避けるため、逆算する温度範囲は各 fit の使用範囲内に限定する。
LINEAR_668_TEMPERATURE_RANGE_K = (3.50, 6.68)
LINEAR_741_TEMPERATURE_RANGE_K = (3.50, 7.41)
MODEL_INVERSION_TEMPERATURE_RANGE_K = (
    float(np.min(temperature_k)),
    float(np.max(temperature_k)),
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "20260527" / "s21_sweep_baseline_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "fr_vs_temperature_linear_and_model_fits_with_inverse_temperature.png"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class InversePrediction:
    """Temperature obtained from a specified resonance frequency."""

    fit_name: str
    target_fr_hz: float
    target_fr_se_hz: float
    temperature_k: float
    temperature_se_k: float
    temperature_se_from_target_fr_k: float
    temperature_se_from_fit_params_k: float


# =============================================================================
# FIT FUNCTIONS
# =============================================================================

def weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Weighted fit of y = intercept + slope*x.

    The covariance is scaled by reduced chi^2.  Thus the reported parameter
    uncertainty includes both the supplied individual y errors and the scatter
    around a straight line.
    """
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(1.0 / yerr**2)

    cov_formal = np.linalg.inv(X.T @ W @ X)
    beta = cov_formal @ X.T @ W @ y
    yfit = X @ beta

    dof = len(x) - 2
    chi2 = np.sum(((y - yfit) / yerr) ** 2)
    covariance = cov_formal * (chi2 / dof)

    intercept_hz, slope_hz_per_k = beta
    intercept_se_hz, slope_se_hz_per_k = np.sqrt(np.diag(covariance))

    return {
        "intercept_hz": float(intercept_hz),
        "slope_hz_per_k": float(slope_hz_per_k),
        "intercept_se_hz": float(intercept_se_hz),
        "slope_se_hz_per_k": float(slope_se_hz_per_k),
        "covariance": covariance,
        "chi2": float(chi2),
        "reduced_chi2": float(chi2 / dof),
    }


def resonator_temperature_model(
    temperature_k: np.ndarray | float,
    fr0_hz: float,
    alpha: float,
    tc_k: float,
) -> np.ndarray:
    """Two-fluid-like temperature dependence model, valid for T < Tc.

        f_r(T) = f_r^0 / sqrt(1 - alpha + alpha / (1 - (T / Tc)^4))
    """
    temperature_k = np.asarray(temperature_k, dtype=float)
    denominator = 1.0 - alpha + alpha / (1.0 - (temperature_k / tc_k) ** 4)
    return fr0_hz / np.sqrt(denominator)


def fit_temperature_model(
    temperature_k: np.ndarray,
    fr_hz: np.ndarray,
    fr_se_hz: np.ndarray,
    use_weights: bool = False,
) -> dict[str, float | np.ndarray]:
    """Nonlinear least-squares fit of the resonator temperature model."""
    tc_lower_bound = float(np.max(temperature_k) * 1.0001)

    p0 = [
        float(np.max(fr_hz) * 1.0005),
        0.05,
        9.5,
    ]
    lower_bounds = [5.0e9, 1.0e-5, tc_lower_bound]
    upper_bounds = [6.0e9, 0.95, 100.0]

    curve_fit_kwargs: dict[str, object] = {
        "p0": p0,
        "bounds": (lower_bounds, upper_bounds),
        "maxfev": 100000,
    }
    if use_weights:
        curve_fit_kwargs["sigma"] = fr_se_hz
        curve_fit_kwargs["absolute_sigma"] = True

    popt, pcov = curve_fit(
        resonator_temperature_model,
        temperature_k,
        fr_hz,
        **curve_fit_kwargs,
    )

    fr0_hz, alpha, tc_k = popt
    fr0_se_hz, alpha_se, tc_se_k = np.sqrt(np.diag(pcov))

    fitted_fr_hz = resonator_temperature_model(temperature_k, *popt)
    residual_hz = fr_hz - fitted_fr_hz
    rss_hz2 = np.sum(residual_hz**2)
    dof = len(temperature_k) - len(popt)

    if use_weights:
        chi2 = np.sum((residual_hz / fr_se_hz) ** 2)
        reduced_chi2 = chi2 / dof
    else:
        chi2 = np.nan
        reduced_chi2 = np.nan

    return {
        "fr0_hz": float(fr0_hz),
        "fr0_se_hz": float(fr0_se_hz),
        "alpha": float(alpha),
        "alpha_se": float(alpha_se),
        "tc_k": float(tc_k),
        "tc_se_k": float(tc_se_k),
        "covariance": pcov,
        "rss_hz2": float(rss_hz2),
        "chi2": float(chi2),
        "reduced_chi2": float(reduced_chi2),
    }


# =============================================================================
# INVERSE FUNCTIONS AND ERROR PROPAGATION
# =============================================================================

def _is_within_model_range(
    target_fr_hz: float,
    model_values_hz: tuple[float, float],
) -> bool:
    """True if target_fr_hz is bracketed by two endpoint model values."""
    lower_hz, upper_hz = sorted(model_values_hz)
    return lower_hz <= target_fr_hz <= upper_hz


def predict_temperature_from_linear_fit(
    target_fr_hz: float,
    target_fr_se_hz: float,
    fit: dict[str, float | np.ndarray],
    temperature_range_k: tuple[float, float],
    fit_name: str,
) -> InversePrediction | None:
    """Invert f_r(T) = a + b T and propagate uncertainty to T.

    For T = (f_target - a) / b,

        sigma_T^2 = (dT/df_target)^2 sigma_f^2
                    + grad_(a,b)(T) C_(a,b) grad_(a,b)(T)^T,

    where grad_(a,b)(T) = (-1/b, -T/b).
    """
    intercept_hz = float(fit["intercept_hz"])
    slope_hz_per_k = float(fit["slope_hz_per_k"])
    covariance = np.asarray(fit["covariance"], dtype=float)

    if slope_hz_per_k == 0.0:
        raise ValueError(f"{fit_name}: slope is zero; inverse temperature is undefined.")

    t_low_k, t_high_k = temperature_range_k
    endpoint_frequencies_hz = (
        intercept_hz + slope_hz_per_k * t_low_k,
        intercept_hz + slope_hz_per_k * t_high_k,
    )
    if not _is_within_model_range(target_fr_hz, endpoint_frequencies_hz):
        return None

    predicted_temperature_k = (target_fr_hz - intercept_hz) / slope_hz_per_k

    # Input-frequency contribution: sigma_T = |dT/df| sigma_f = sigma_f / |b|.
    sigma_from_target_fr_k = abs(target_fr_se_hz / slope_hz_per_k)

    # Fit-parameter contribution, including the a-b covariance.
    parameter_gradient = np.array([
        -1.0 / slope_hz_per_k,
        -predicted_temperature_k / slope_hz_per_k,
    ])
    variance_from_fit_params_k2 = parameter_gradient @ covariance @ parameter_gradient
    sigma_from_fit_params_k = float(np.sqrt(max(variance_from_fit_params_k2, 0.0)))

    sigma_total_k = float(np.hypot(sigma_from_target_fr_k, sigma_from_fit_params_k))

    return InversePrediction(
        fit_name=fit_name,
        target_fr_hz=target_fr_hz,
        target_fr_se_hz=target_fr_se_hz,
        temperature_k=float(predicted_temperature_k),
        temperature_se_k=sigma_total_k,
        temperature_se_from_target_fr_k=float(sigma_from_target_fr_k),
        temperature_se_from_fit_params_k=sigma_from_fit_params_k,
    )


def model_partials(
    temperature_k: float,
    fr0_hz: float,
    alpha: float,
    tc_k: float,
) -> tuple[float, np.ndarray]:
    """Return df/dT and (df/dfr0, df/dalpha, df/dTc) analytically."""
    r = (temperature_k / tc_k) ** 4
    one_minus_r = 1.0 - r
    denominator = 1.0 - alpha + alpha / one_minus_r
    f_hz = fr0_hz / np.sqrt(denominator)

    ddenominator_dtemperature = (
        alpha
        * (4.0 * temperature_k**3 / tc_k**4)
        / one_minus_r**2
    )
    df_dtemperature = -0.5 * fr0_hz * denominator ** (-1.5) * ddenominator_dtemperature

    ddenominator_dalpha = -1.0 + 1.0 / one_minus_r
    ddenominator_dtc = (
        alpha
        * (-4.0 * temperature_k**4 / tc_k**5)
        / one_minus_r**2
    )

    df_dfr0 = 1.0 / np.sqrt(denominator)
    df_dalpha = -0.5 * fr0_hz * denominator ** (-1.5) * ddenominator_dalpha
    df_dtc = -0.5 * fr0_hz * denominator ** (-1.5) * ddenominator_dtc

    return float(df_dtemperature), np.array([df_dfr0, df_dalpha, df_dtc])


def predict_temperature_from_nonlinear_model(
    target_fr_hz: float,
    target_fr_se_hz: float,
    fit: dict[str, float | np.ndarray],
    temperature_range_k: tuple[float, float],
    fit_name: str,
) -> InversePrediction | None:
    """Numerically invert f_r(T) and propagate both input and fit errors.

    The inverse-function contribution from the specified frequency is

        sigma_T,fr = |dT/df| sigma_fr = sigma_fr / |df/dT|.

    The fit-parameter part follows the implicit relation f(T, p) = f_target:

        dT/dp_i = - (df/dp_i) / (df/dT),

    and uses the full curve_fit covariance matrix C_p:

        sigma_T,param^2 = grad_p(T) C_p grad_p(T)^T.
    """
    fr0_hz = float(fit["fr0_hz"])
    alpha = float(fit["alpha"])
    tc_k = float(fit["tc_k"])
    covariance = np.asarray(fit["covariance"], dtype=float)

    temp_low_k, temp_high_k = temperature_range_k
    if not (0.0 <= temp_low_k < temp_high_k < tc_k):
        raise ValueError(
            "MODEL_INVERSION_TEMPERATURE_RANGE_K must satisfy "
            "0 <= T_low < T_high < fitted Tc."
        )

    def residual(temp_k: float) -> float:
        return float(resonator_temperature_model(temp_k, fr0_hz, alpha, tc_k) - target_fr_hz)

    endpoint_frequencies_hz = (
        float(resonator_temperature_model(temp_low_k, fr0_hz, alpha, tc_k)),
        float(resonator_temperature_model(temp_high_k, fr0_hz, alpha, tc_k)),
    )
    if not _is_within_model_range(target_fr_hz, endpoint_frequencies_hz):
        return None

    predicted_temperature_k = float(brentq(residual, temp_low_k, temp_high_k))
    df_dtemperature, df_dparams = model_partials(
        predicted_temperature_k,
        fr0_hz,
        alpha,
        tc_k,
    )
    if df_dtemperature == 0.0:
        raise ValueError(f"{fit_name}: df/dT is zero; inverse uncertainty is undefined.")

    sigma_from_target_fr_k = abs(target_fr_se_hz / df_dtemperature)
    parameter_gradient = -df_dparams / df_dtemperature
    variance_from_fit_params_k2 = parameter_gradient @ covariance @ parameter_gradient
    sigma_from_fit_params_k = float(np.sqrt(max(variance_from_fit_params_k2, 0.0)))
    sigma_total_k = float(np.hypot(sigma_from_target_fr_k, sigma_from_fit_params_k))

    return InversePrediction(
        fit_name=fit_name,
        target_fr_hz=target_fr_hz,
        target_fr_se_hz=target_fr_se_hz,
        temperature_k=predicted_temperature_k,
        temperature_se_k=sigma_total_k,
        temperature_se_from_target_fr_k=float(sigma_from_target_fr_k),
        temperature_se_from_fit_params_k=sigma_from_fit_params_k,
    )


# =============================================================================
# FIT
# =============================================================================

# Note: temperature_k is stored in descending order.  Use a mask so that the
# 3.50–6.68 K fit really excludes the 7.41 K datum.
mask_668 = (
    (temperature_k >= LINEAR_668_TEMPERATURE_RANGE_K[0])
    & (temperature_k <= LINEAR_668_TEMPERATURE_RANGE_K[1])
)

fit_to_668 = weighted_linear_fit(
    temperature_k[mask_668],
    fr_hz[mask_668],
    fr_se_hz[mask_668],
)
fit_to_741 = weighted_linear_fit(temperature_k, fr_hz, fr_se_hz)
model_fit = fit_temperature_model(
    temperature_k,
    fr_hz,
    fr_se_hz,
    use_weights=MODEL_FIT_USE_WEIGHTS,
)

fit_method_label = (
    "Weighted nonlinear least squares"
    if MODEL_FIT_USE_WEIGHTS
    else "Nonlinear least squares"
)

# Check user input before carrying out inverse predictions.
target_fr_ghz_array = np.asarray(TARGET_FR_GHZ, dtype=float)
target_fr_se_hz_array = np.asarray(TARGET_FR_SE_HZ, dtype=float)
if target_fr_ghz_array.ndim != 1 or target_fr_se_hz_array.ndim != 1:
    raise ValueError("TARGET_FR_GHZ and TARGET_FR_SE_HZ must both be one-dimensional lists.")
if len(target_fr_ghz_array) != len(target_fr_se_hz_array):
    raise ValueError("TARGET_FR_GHZ and TARGET_FR_SE_HZ must have the same length.")
if np.any(target_fr_se_hz_array < 0.0):
    raise ValueError("TARGET_FR_SE_HZ must contain non-negative values.")

# Invert every fit for every specified frequency.
inverse_predictions: list[InversePrediction] = []
for target_fr_ghz, target_fr_se_hz in zip(target_fr_ghz_array, target_fr_se_hz_array):
    target_fr_hz = float(target_fr_ghz * 1e9)

    candidates = [
        predict_temperature_from_linear_fit(
            target_fr_hz,
            float(target_fr_se_hz),
            fit_to_668,
            LINEAR_668_TEMPERATURE_RANGE_K,
            "Weighted linear fit: 3.50–6.68 K",
        ),
        predict_temperature_from_linear_fit(
            target_fr_hz,
            float(target_fr_se_hz),
            fit_to_741,
            LINEAR_741_TEMPERATURE_RANGE_K,
            "Weighted linear fit: 3.50–7.41 K",
        ),
        predict_temperature_from_nonlinear_model(
            target_fr_hz,
            float(target_fr_se_hz),
            model_fit,
            MODEL_INVERSION_TEMPERATURE_RANGE_K,
            f"{fit_method_label}: temperature model",
        ),
    ]

    for prediction in candidates:
        if prediction is None:
            print(
                f"warning: f_r = {target_fr_ghz:.6f} GHz cannot be inverted "
                "inside the valid temperature range of one fit; skipped for that fit."
            )
        else:
            inverse_predictions.append(prediction)


# =============================================================================
# PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(11.3, 7.4))

# Measured resonance frequencies
ax.errorbar(
    temperature_k,
    fr_hz / 1e9,
    yerr=fr_se_hz / 1e9,
    fmt="o",
    ms=5,
    capsize=3,
    lw=1.1,
    label=r"Resonance-fit result ($f_r \pm \sigma_{f_r}$)",
)

# Linear fit: 3.50–6.68 K
x_668 = np.linspace(*LINEAR_668_TEMPERATURE_RANGE_K, 300)
y_668_ghz = (
    float(fit_to_668["intercept_hz"])
    + float(fit_to_668["slope_hz_per_k"]) * x_668
) / 1e9
ax.plot(
    x_668,
    y_668_ghz,
    lw=2.0,
    label=(
        "Weighted linear fit: 3.50–6.68 K\n"
        rf"$f_r(T)=({float(fit_to_668['intercept_hz'])/1e9:.4f}"
        rf"\pm{float(fit_to_668['intercept_se_hz'])/1e9:.4f})$ GHz"
        "\n"
        rf"$+({float(fit_to_668['slope_hz_per_k'])/1e6:.2f}"
        rf"\pm{float(fit_to_668['slope_se_hz_per_k'])/1e6:.2f})$ MHz/K $\times T$"
    ),
)

# Linear fit: 3.50–7.41 K
x_741 = np.linspace(*LINEAR_741_TEMPERATURE_RANGE_K, 300)
y_741_ghz = (
    float(fit_to_741["intercept_hz"])
    + float(fit_to_741["slope_hz_per_k"]) * x_741
) / 1e9
ax.plot(
    x_741,
    y_741_ghz,
    "--",
    lw=2.0,
    label=(
        "Weighted linear fit: 3.50–7.41 K\n"
        rf"$f_r(T)=({float(fit_to_741['intercept_hz'])/1e9:.4f}"
        rf"\pm{float(fit_to_741['intercept_se_hz'])/1e9:.4f})$ GHz"
        "\n"
        rf"$+({float(fit_to_741['slope_hz_per_k'])/1e6:.2f}"
        rf"\pm{float(fit_to_741['slope_se_hz_per_k'])/1e6:.2f})$ MHz/K $\times T$"
    ),
)

# Nonlinear temperature-model fit
x_model = np.linspace(np.min(temperature_k), np.max(temperature_k), 600)
y_model_ghz = resonator_temperature_model(
    x_model,
    float(model_fit["fr0_hz"]),
    float(model_fit["alpha"]),
    float(model_fit["tc_k"]),
) / 1e9
model_label = (
    f"{fit_method_label}: temperature model\n"
    rf"$f_r^0 = ({float(model_fit['fr0_hz'])/1e9:.5f}"
    rf"\pm{float(model_fit['fr0_se_hz'])/1e9:.5f})$ GHz"
    "\n"
    rf"$\alpha = {float(model_fit['alpha']):.4f}"
    rf"\pm{float(model_fit['alpha_se']):.4f}$"
    "\n"
    rf"$T_c = ({float(model_fit['tc_k']):.3f}"
    rf"\pm{float(model_fit['tc_se_k']):.3f})$ K"
)
ax.plot(x_model, y_model_ghz, "-.", lw=2.4, label=model_label)

# Mark inverse temperatures from all three fitted functions.
marker_by_fit = {
    "Weighted linear fit: 3.50–6.68 K": "o",
    "Weighted linear fit: 3.50–7.41 K": "s",
    f"{fit_method_label}: temperature model": "*",
}
short_label_by_fit = {
    "Weighted linear fit: 3.50–6.68 K": "Inverse T: linear 3.50–6.68 K",
    "Weighted linear fit: 3.50–7.41 K": "Inverse T: linear 3.50–7.41 K",
    f"{fit_method_label}: temperature model": "Inverse T: temperature model",
}
# 各 fit の凡例に、指定周波数から逆算した温度も書く
legend_label_by_fit: dict[str, str] = {}

for fit_name, short_name in short_label_by_fit.items():
    predictions_for_this_fit = [
        p for p in inverse_predictions
        if p.fit_name == fit_name
    ]

    if not predictions_for_this_fit:
        continue

    inverse_temperature_lines = "\n".join(
        rf"$f_r={p.target_fr_hz / 1e9:.6f}\,\mathrm{{GHz}}"
        rf"\ \Rightarrow\ "
        rf"T={p.temperature_k:.3f}"
        rf"\pm{p.temperature_se_k:.3f}\,\mathrm{{K}}$"
        for p in predictions_for_this_fit
    )

    legend_label_by_fit[fit_name] = (
        f"{short_name}\n"
        f"{inverse_temperature_lines}"
    )

shown_fit_labels: set[str] = set()

for index, prediction in enumerate(inverse_predictions):
    if prediction.fit_name in shown_fit_labels:
        label = "_nolegend_"
    else:
        shown_fit_labels.add(prediction.fit_name)
        label = legend_label_by_fit[prediction.fit_name]

    ax.errorbar(
        prediction.temperature_k,
        prediction.target_fr_hz / 1e9,
        xerr=prediction.temperature_se_k,
        fmt=marker_by_fit[prediction.fit_name],
        ms=8 if marker_by_fit[prediction.fit_name] == "*" else 5.5,
        capsize=3,
        lw=1.1,
        zorder=7,
        label=label,
    )

# Draw one horizontal guide per target frequency and annotate it at the model point.
for target_index, target_fr_ghz in enumerate(target_fr_ghz_array):
    ax.axhline(target_fr_ghz, lw=0.8, alpha=0.35)
    model_prediction = next(
        (
            p for p in inverse_predictions
            if np.isclose(p.target_fr_hz, target_fr_ghz * 1e9)
            and p.fit_name == f"{fit_method_label}: temperature model"
        ),
        None,
    )
    if model_prediction is not None:
        ax.annotate(
            rf"$f_r={target_fr_ghz:.6f}$ GHz"
            "\n"
            rf"model: $T={model_prediction.temperature_k:.3f}"
            rf"\pm{model_prediction.temperature_se_k:.3f}$ K",
            xy=(model_prediction.temperature_k, target_fr_ghz),
            xytext=(10, 9 + 30 * target_index),
            textcoords="offset points",
            fontsize=8.1,
            ha="left",
            va="bottom",
        )

ax.set_xlabel("Temperature [K]")
ax.set_ylabel(r"Resonance frequency $f_r$ [GHz]")
ax.set_title("Temperature dependence of resonance frequency")
ax.grid(True, alpha=0.30)
ax.legend(loc="best", fontsize=7.3)
ax.ticklabel_format(axis="y", style="plain", useOffset=False)

fig.tight_layout()
fig.savefig(OUT_FILE, dpi=250, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# TERMINAL OUTPUT
# =============================================================================

print(f"saved: {OUT_FILE}")
print()
print("=== Temperature-model fit result ===")
print(f"fit method = {fit_method_label}")
print(
    f"fr0 = {float(model_fit['fr0_hz'])/1e9:.7f} "
    f"+/- {float(model_fit['fr0_se_hz'])/1e9:.7f} GHz"
)
print(
    f"alpha = {float(model_fit['alpha']):.7f} "
    f"+/- {float(model_fit['alpha_se']):.7f}"
)
print(
    f"Tc = {float(model_fit['tc_k']):.7f} "
    f"+/- {float(model_fit['tc_se_k']):.7f} K"
)
print(f"RSS = {float(model_fit['rss_hz2']):.4e} Hz^2")

if inverse_predictions:
    print()
    print("=== Specified-frequency inverse temperature ===")
    print("The total error combines the target-frequency and fit-parameter terms.")
    for prediction in inverse_predictions:
        print(
            f"{prediction.fit_name}\n"
            f"  f_r = {prediction.target_fr_hz / 1e9:.6f} "
            f"+/- {prediction.target_fr_se_hz:.3f} Hz\n"
            f"  T = {prediction.temperature_k:.6f} "
            f"+/- {prediction.temperature_se_k:.6f} K\n"
            f"    from target f_r: +/- {prediction.temperature_se_from_target_fr_k:.6f} K\n"
            f"    from fit parameters: +/- {prediction.temperature_se_from_fit_params_k:.6f} K"
        )

if MODEL_FIT_USE_WEIGHTS:
    print(f"chi2 / dof = {float(model_fit['reduced_chi2']):.3f}")
