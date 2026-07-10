from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


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

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "20260527" / "s21_sweep_baseline_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "fr_vs_temperature_linear_and_model_fits.png"


# =============================================================================
# FIT FUNCTIONS
# =============================================================================

def weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
) -> dict[str, float]:
    """Weighted fit of y = intercept + slope*x."""
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
        "intercept_hz": intercept_hz,
        "slope_hz_per_k": slope_hz_per_k,
        "intercept_se_hz": intercept_se_hz,
        "slope_se_hz_per_k": slope_se_hz_per_k,
    }


def resonator_temperature_model(
    temperature_k: np.ndarray,
    fr0_hz: float,
    alpha: float,
    tc_k: float,
) -> np.ndarray:
    """
    Two-fluid-like temperature dependence model:

        f_r(T) = f_r^0 /
                 sqrt(1 - alpha + alpha / (1 - (T / Tc)^4))

    This expression is valid for T < Tc.
    """
    denominator = 1.0 - alpha + alpha / (1.0 - (temperature_k / tc_k) ** 4)
    return fr0_hz / np.sqrt(denominator)


def fit_temperature_model(
    temperature_k: np.ndarray,
    fr_hz: np.ndarray,
    fr_se_hz: np.ndarray,
    use_weights: bool = False,
) -> dict[str, float]:
    """
    Nonlinear least-squares fit of the resonator temperature model.

    Free parameters:
        fr0_hz : zero-temperature resonance frequency
        alpha  : kinetic inductance fraction
        tc_k   : critical temperature
    """

    # Tc must be larger than every measured temperature
    tc_lower_bound = float(np.max(temperature_k) * 1.0001)

    # Initial parameters:
    # fr0 is slightly above the maximum observed resonance frequency.
    p0 = [
        float(np.max(fr_hz) * 1.0005),
        0.05,
        9.5,
    ]

    lower_bounds = [
        5.0e9,          # fr0 [Hz]
        1.0e-5,         # alpha
        tc_lower_bound, # Tc [K]
    ]
    upper_bounds = [
        6.0e9,          # fr0 [Hz]
        0.95,           # alpha
        100.0,          # Tc [K]
    ]

    curve_fit_kwargs = {
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

    fitted_fr_hz = resonator_temperature_model(
        temperature_k,
        fr0_hz,
        alpha,
        tc_k,
    )

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
        "fr0_hz": fr0_hz,
        "fr0_se_hz": fr0_se_hz,
        "alpha": alpha,
        "alpha_se": alpha_se,
        "tc_k": tc_k,
        "tc_se_k": tc_se_k,
        "rss_hz2": rss_hz2,
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
    }


# =============================================================================
# FIT
# =============================================================================

fit_to_668 = weighted_linear_fit(
    temperature_k[:5],
    fr_hz[:5],
    fr_se_hz[:5],
)

fit_to_741 = weighted_linear_fit(
    temperature_k,
    fr_hz,
    fr_se_hz,
)

model_fit = fit_temperature_model(
    temperature_k[0:],
    fr_hz[0:],
    fr_se_hz[0:],
    use_weights=MODEL_FIT_USE_WEIGHTS,
)


# =============================================================================
# PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(10.5, 7.0))

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

# Linear fit: 3.50--6.68 K
x_668 = np.linspace(3.50, 6.68, 300)
y_668_ghz = (
    fit_to_668["intercept_hz"]
    + fit_to_668["slope_hz_per_k"] * x_668
) / 1e9

ax.plot(
    x_668,
    y_668_ghz,
    lw=2.0,
    label=(
        "Weighted linear fit: 3.50–6.68 K\n"
        rf"$f_r(T)=({fit_to_668['intercept_hz']/1e9:.4f}"
        rf"\pm{fit_to_668['intercept_se_hz']/1e9:.4f})$ GHz"
        "\n"
        rf"$+({fit_to_668['slope_hz_per_k']/1e6:.2f}"
        rf"\pm{fit_to_668['slope_se_hz_per_k']/1e6:.2f})$ MHz/K $\times T$"
    ),
)

# Linear fit: 3.50--7.41 K
x_741 = np.linspace(3.50, 7.41, 300)
y_741_ghz = (
    fit_to_741["intercept_hz"]
    + fit_to_741["slope_hz_per_k"] * x_741
) / 1e9

ax.plot(
    x_741,
    y_741_ghz,
    "--",
    lw=2.0,
    label=(
        "Weighted linear fit: 3.50–7.41 K\n"
        rf"$f_r(T)=({fit_to_741['intercept_hz']/1e9:.4f}"
        rf"\pm{fit_to_741['intercept_se_hz']/1e9:.4f})$ GHz"
        "\n"
        rf"$+({fit_to_741['slope_hz_per_k']/1e6:.2f}"
        rf"\pm{fit_to_741['slope_se_hz_per_k']/1e6:.2f})$ MHz/K $\times T$"
    ),
)

# Nonlinear temperature-model fit
x_model = np.linspace(
    np.min(temperature_k),
    np.max(temperature_k),
    600,
)

y_model_ghz = resonator_temperature_model(
    x_model,
    model_fit["fr0_hz"],
    model_fit["alpha"],
    model_fit["tc_k"],
) / 1e9

fit_method_label = (
    "Weighted nonlinear least squares"
    if MODEL_FIT_USE_WEIGHTS
    else "Nonlinear least squares"
)

model_label = (
    f"{fit_method_label}: temperature model\n"
    rf"$f_r^0 = ({model_fit['fr0_hz']/1e9:.5f}"
    rf"\pm{model_fit['fr0_se_hz']/1e9:.5f})$ GHz"
    "\n"
    rf"$\alpha = {model_fit['alpha']:.4f}"
    rf"\pm{model_fit['alpha_se']:.4f}$"
    "\n"
    rf"$T_c = ({model_fit['tc_k']:.3f}"
    rf"\pm{model_fit['tc_se_k']:.3f})$ K"
)

ax.plot(
    x_model,
    y_model_ghz,
    "-.",
    lw=2.4,
    label=model_label,
)

ax.set_xlabel("Temperature [K]")
ax.set_ylabel(r"Resonance frequency $f_r$ [GHz]")
ax.set_title("Temperature dependence of resonance frequency")
ax.grid(True, alpha=0.30)
ax.legend(loc="best", fontsize=7.8)
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
    f"fr0 = {model_fit['fr0_hz']/1e9:.7f} "
    f"+/- {model_fit['fr0_se_hz']/1e9:.7f} GHz"
)
print(
    f"alpha = {model_fit['alpha']:.7f} "
    f"+/- {model_fit['alpha_se']:.7f}"
)
print(
    f"Tc = {model_fit['tc_k']:.7f} "
    f"+/- {model_fit['tc_se_k']:.7f} K"
)
print(f"RSS = {model_fit['rss_hz2']:.4e} Hz^2")

if MODEL_FIT_USE_WEIGHTS:
    print(f"chi2 / dof = {model_fit['reduced_chi2']:.3f}")

# from __future__ import annotations

# from pathlib import Path

# import matplotlib
# matplotlib.use("Agg")

# import matplotlib.pyplot as plt
# import numpy as np


# # Values transcribed from the resonator-fit screenshots:
# # T [K], fr [Hz], standard error of fr [Hz]
# temperature_k = np.array([7.41, 6.68, 6.1, 5.56, 4.74, 3.5])
# fr_hz = np.array([5444033000.0, 5464446000.0, 5473735000.0, 5485420000.0, 5485600000.0, 5489634000.0])
# fr_se_hz = np.array([188348.6, 139376.4, 124655.4, 100795.8, 103458.6, 94674.31])


# def weighted_linear_fit(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> dict[str, float]:
#     """Weighted fit of y = intercept + slope*x.

#     The fit uses 1/yerr^2 weights.  The reported parameter standard
#     errors are scaled by reduced chi^2, so they reflect the scatter
#     around a straight line as well as the individual fr fit errors.
#     """
#     X = np.column_stack([np.ones_like(x), x])
#     W = np.diag(1.0 / yerr**2)

#     cov_formal = np.linalg.inv(X.T @ W @ X)
#     beta = cov_formal @ X.T @ W @ y
#     yfit = X @ beta

#     dof = len(x) - 2
#     chi2 = np.sum(((y - yfit) / yerr) ** 2)
#     covariance = cov_formal * (chi2 / dof)

#     intercept_hz, slope_hz_per_k = beta
#     intercept_se_hz, slope_se_hz_per_k = np.sqrt(np.diag(covariance))

#     return {
#         "intercept_hz": intercept_hz,
#         "slope_hz_per_k": slope_hz_per_k,
#         "intercept_se_hz": intercept_se_hz,
#         "slope_se_hz_per_k": slope_se_hz_per_k,
#     }


# HERE = Path(__file__).resolve().parent
# OUT_DIR = HERE / "data" / "20260527" / "s21_sweep_baseline_compare"
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# OUT_FILE = OUT_DIR / "fr_vs_temperature_linear_fits.png"

# fit_to_668 = weighted_linear_fit(temperature_k[:5], fr_hz[:5], fr_se_hz[:5])
# fit_to_741 = weighted_linear_fit(temperature_k, fr_hz, fr_se_hz)

# fig, ax = plt.subplots(figsize=(9.3, 6.2))

# # Data are points only; they are not connected with lines.
# ax.errorbar(
#     temperature_k,
#     fr_hz / 1e9,
#     yerr=fr_se_hz / 1e9,
#     fmt="o",
#     ms=5,
#     capsize=3,
#     lw=1.1,
#     label=r"Resonance-fit result ($f_r \pm S21fitE$)",
# )

# x_668 = np.linspace(3.50, 6.68, 300)
# y_668_ghz = (
#     fit_to_668["intercept_hz"] + fit_to_668["slope_hz_per_k"] * x_668
# ) / 1e9
# ax.plot(
#     x_668,
#     y_668_ghz,
#     lw=2.0,
#     label=(
#         "Weighted linear fit: 3.50–6.68 K\n"
#         rf"$f_r(T) = ({fit_to_668['intercept_hz']/1e9:.4f}"
#         rf" \pm {fit_to_668['intercept_se_hz']/1e9:.4f})$ GHz"
#         "\n"
#         rf"$\quad + ({fit_to_668['slope_hz_per_k']/1e6:.2f}"
#         rf" \pm {fit_to_668['slope_se_hz_per_k']/1e6:.2f})$ MHz/K $\times T$"
#     ),
# )

# x_741 = np.linspace(3.50, 7.41, 300)
# y_741_ghz = (
#     fit_to_741["intercept_hz"] + fit_to_741["slope_hz_per_k"] * x_741
# ) / 1e9
# ax.plot(
#     x_741,
#     y_741_ghz,
#     "--",
#     lw=2.0,
#     label=(
#         "Weighted linear fit: 3.50–7.41 K\n"
#         rf"$f_r(T) = ({fit_to_741['intercept_hz']/1e9:.4f}"
#         rf" \pm {fit_to_741['intercept_se_hz']/1e9:.4f})$ GHz"
#         "\n"
#         rf"$\quad + ({fit_to_741['slope_hz_per_k']/1e6:.2f}"
#         rf" \pm {fit_to_741['slope_se_hz_per_k']/1e6:.2f})$ MHz/K $\times T$"
#     ),
# )

# ax.set_xlabel("Temperature [K]")
# ax.set_ylabel(r"Resonance frequency $f_r$ [GHz]")
# ax.set_title(r"Temperature dependence of resonance frequency")
# ax.grid(True, alpha=0.30)
# ax.legend(loc="upper left", fontsize=8)
# ax.ticklabel_format(axis="y", style="plain", useOffset=False)

# fig.tight_layout()
# fig.savefig(OUT_FILE, dpi=250, bbox_inches="tight")
# plt.close(fig)

# print(f"saved: {OUT_FILE}")
