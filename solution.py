import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, least_squares


# -----------------------------
# Constants from the assignment
# -----------------------------
Y_SHIFT = 42
T_MIN = 6
T_MAX = 60

THETA_MIN_DEG = 0
THETA_MAX_DEG = 50
M_MIN = -0.05
M_MAX = 0.05
X_MIN = 0
X_MAX = 100


def parametric_curve(t, theta_rad, M, X):
    """
    Generate x and y values from the given parametric equation.

    x = t*cos(theta) - e^(M|t|)*sin(0.3t)*sin(theta) + X
    y = 42 + t*sin(theta) + e^(M|t|)*sin(0.3t)*cos(theta)
    """
    wave = np.exp(M * np.abs(t)) * np.sin(0.3 * t)

    x = t * np.cos(theta_rad) - wave * np.sin(theta_rad) + X
    y = Y_SHIFT + t * np.sin(theta_rad) + wave * np.cos(theta_rad)

    return x, y


def inverse_transform(x, y, theta_rad, X):
    """
    Reverse the rotation and horizontal shift.

    If theta and X are correct:
    u should be approximately equal to t
    v should be approximately equal to e^(M|t|)*sin(0.3t)
    """
    u = (x - X) * np.cos(theta_rad) + (y - Y_SHIFT) * np.sin(theta_rad)
    v = -(x - X) * np.sin(theta_rad) + (y - Y_SHIFT) * np.cos(theta_rad)

    return u, v


def residuals(params, x_actual, y_actual):
    """
    Residual function used for optimization.

    params = [theta_degrees, M, X]
    """
    theta_deg, M, X = params
    theta_rad = np.deg2rad(theta_deg)

    u, v = inverse_transform(x_actual, y_actual, theta_rad, X)

    expected_v = np.exp(M * np.abs(u)) * np.sin(0.3 * u)

    curve_error = v - expected_v

    lower_penalty = np.maximum(T_MIN - u, 0)
    upper_penalty = np.maximum(u - T_MAX, 0)

    return np.concatenate([
        curve_error,
        10 * lower_penalty,
        10 * upper_penalty
    ])


def objective(params, x_actual, y_actual):
    """
    L1-style objective function.
    Lower value means better fit.
    """
    r = residuals(params, x_actual, y_actual)
    return np.mean(np.abs(r))


def fit_parameters(x_actual, y_actual):
    """
    First performs global optimization using differential evolution,
    then refines the result using least squares.
    """
    bounds = [
        (THETA_MIN_DEG, THETA_MAX_DEG),
        (M_MIN, M_MAX),
        (X_MIN, X_MAX)
    ]

    global_result = differential_evolution(
        objective,
        bounds=bounds,
        args=(x_actual, y_actual),
        seed=42,
        tol=1e-10,
        polish=True,
        workers=1
    )

    refined_result = least_squares(
        residuals,
        global_result.x,
        args=(x_actual, y_actual),
        bounds=(
            [THETA_MIN_DEG, M_MIN, X_MIN],
            [THETA_MAX_DEG, M_MAX, X_MAX]
        ),
        max_nfev=5000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12
    )

    theta_deg, M, X = refined_result.x
    theta_rad = np.deg2rad(theta_deg)

    return theta_deg, theta_rad, M, X


def calculate_predicted_points_for_actual_data(x_actual, y_actual, theta_rad, M, X):
    """
    Recover estimated t values from actual data, then generate predicted points
    at those same estimated t values.
    """
    t_estimated, _ = inverse_transform(x_actual, y_actual, theta_rad, X)
    x_predicted, y_predicted = parametric_curve(t_estimated, theta_rad, M, X)

    return t_estimated, x_predicted, y_predicted


def calculate_l1_error(x_actual, y_actual, x_predicted, y_predicted):
    """
    Calculates point-wise L1 error.
    """
    l1_per_point = np.abs(x_actual - x_predicted) + np.abs(y_actual - y_predicted)
    mean_l1_error = np.mean(l1_per_point)

    return mean_l1_error, l1_per_point


def create_output_plots(x_actual, y_actual, theta_rad, M, X):
    """
    Creates multiple plots.

    Important:
    - All CSV points are used for fitting and error calculation.
    - Only the overlay visualization uses a reduced number of actual points
      so that the predicted curve remains visible.
    """
    os.makedirs("outputs", exist_ok=True)

    t_estimated, x_predicted_actual_t, y_predicted_actual_t = (
        calculate_predicted_points_for_actual_data(x_actual, y_actual, theta_rad, M, X)
    )

    mean_l1_error, l1_per_point = calculate_l1_error(
        x_actual,
        y_actual,
        x_predicted_actual_t,
        y_predicted_actual_t
    )

    t_curve = np.linspace(T_MIN, T_MAX, 1500)
    x_curve, y_curve = parametric_curve(t_curve, theta_rad, M, X)

    sort_idx = np.argsort(t_estimated)

    # ----------------------------------------------------
    # Sampling only for visualization
    # This does NOT affect fitting or error calculation.
    # ----------------------------------------------------
    sample_step = 20
    sampled_indices = np.arange(0, len(x_actual), sample_step)

    x_sampled = x_actual[sampled_indices]
    y_sampled = y_actual[sampled_indices]

    # ----------------------------------------------------
    # Plot 1: Better overlay plot
    # Shows full predicted curve + sampled actual points
    # ----------------------------------------------------
    plt.figure(figsize=(10, 7))

    plt.plot(
        x_curve,
        y_curve,
        linewidth=2.8,
        label="Predicted Curve",
        zorder=2
    )

    plt.scatter(
        x_sampled,
        y_sampled,
        s=45,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=1.2,
        label=f"Actual CSV Points - every {sample_step}th point",
        zorder=3
    )

    plt.title("Predicted Curve vs Sampled Actual CSV Points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig("outputs/curve_comparison_overlay.png", dpi=300)
    plt.close()

    # ----------------------------------------------------
    # Plot 2: Side-by-side comparison
    # Uses sampled actual points for cleaner visual comparison
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(
        x_sampled,
        y_sampled,
        s=35,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=1
    )
    axes[0].set_title(f"Actual CSV Points - every {sample_step}th point")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].plot(
        x_curve,
        y_curve,
        linewidth=2.5
    )
    axes[1].set_title("Predicted Curve")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal", adjustable="box")

    fig.suptitle("Side-by-Side Curve Comparison", fontsize=14)
    fig.tight_layout()
    plt.savefig("outputs/curve_comparison_side_by_side.png", dpi=300)
    plt.close()

    # ----------------------------------------------------
    # Plot 3: Residual analysis
    # Uses all points because this is a mathematical validation.
    # ----------------------------------------------------
    x_error = x_actual - x_predicted_actual_t
    y_error = y_actual - y_predicted_actual_t

    plt.figure(figsize=(11, 7))

    plt.plot(
        t_estimated[sort_idx],
        x_error[sort_idx],
        linewidth=1.5,
        label="x residual"
    )

    plt.plot(
        t_estimated[sort_idx],
        y_error[sort_idx],
        linewidth=1.5,
        label="y residual"
    )

    plt.plot(
        t_estimated[sort_idx],
        l1_per_point[sort_idx],
        linewidth=1.5,
        label="L1 error per point"
    )

    plt.axhline(0, linewidth=1, linestyle="--")

    plt.title(f"Residual Analysis | Mean L1 Error = {mean_l1_error:.12f}")
    plt.xlabel("Estimated t")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/residual_analysis.png", dpi=300)
    plt.close()

    # ----------------------------------------------------
    # Plot 4: Residual scatter
    # Uses all points.
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.scatter(
        x_error,
        y_error,
        s=12,
        alpha=0.7
    )

    plt.axhline(0, linewidth=1, linestyle="--")
    plt.axvline(0, linewidth=1, linestyle="--")

    plt.title("Residual Scatter Plot: x Error vs y Error")
    plt.xlabel("x error")
    plt.ylabel("y error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/residual_scatter.png", dpi=300)
    plt.close()

    return mean_l1_error    


def save_results(theta_deg, theta_rad, M, X, l1_error):
    """
    Saves final parameter values and equation to a text file.
    """
    os.makedirs("outputs", exist_ok=True)

    equation = (
        rf"\left("
        rf"t\cos({theta_rad:.6f})"
        rf"-e^{{{M:.6f}\left|t\right|}}\sin(0.3t)\sin({theta_rad:.6f})"
        rf"+{X:.6f},\ "
        rf"42+t\sin({theta_rad:.6f})"
        rf"+e^{{{M:.6f}\left|t\right|}}\sin(0.3t)\cos({theta_rad:.6f})"
        rf"\right)"
    )

    result_text = f"""
Final Parameters
----------------
theta = {theta_deg:.10f} degrees
theta = {theta_rad:.10f} radians
M     = {M:.10f}
X     = {X:.10f}

t range = {T_MIN} to {T_MAX}

Mean L1 Error
-------------
{l1_error:.12f}

Final Equation for README / Desmos
----------------------------------
{equation}

Generated Output Files
----------------------
outputs/curve_comparison_overlay.png
outputs/curve_comparison_side_by_side.png
outputs/residual_analysis.png
outputs/residual_scatter.png
outputs/final_result.txt

Note
----
All CSV points were used for fitting and L1 error calculation.
For the overlay plot, only every 20th CSV point was displayed so that the predicted curve remains visible.
"""

    with open("outputs/final_result.txt", "w", encoding="utf-8") as file:
        file.write(result_text)

    return equation


def main():
    csv_path = "xy_data.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "xy_data.csv not found. Please keep xy_data.csv in the same folder as solution.py."
        )

    data = pd.read_csv(csv_path)

    if "x" not in data.columns or "y" not in data.columns:
        raise ValueError("CSV file must contain columns named 'x' and 'y'.")

    x_actual = data["x"].to_numpy(dtype=float)
    y_actual = data["y"].to_numpy(dtype=float)

    theta_deg, theta_rad, M, X = fit_parameters(x_actual, y_actual)

    l1_error = create_output_plots(x_actual, y_actual, theta_rad, M, X)

    equation = save_results(theta_deg, theta_rad, M, X, l1_error)

    print("\nFinal Parameters")
    print("----------------")
    print(f"theta = {theta_deg:.10f} degrees")
    print(f"theta = {theta_rad:.10f} radians")
    print(f"M     = {M:.10f}")
    print(f"X     = {X:.10f}")
    print(f"L1 error = {l1_error:.12f}")

    print("\nFinal Equation for README / Desmos")
    print("----------------------------------")
    print(equation)

    print("\nFiles generated:")
    print("outputs/curve_comparison_overlay.png")
print("outputs/curve_comparison_side_by_side.png")
print("outputs/residual_analysis.png")
print("outputs/residual_scatter.png")
print("outputs/final_result.txt")


if __name__ == "__main__":
    main()