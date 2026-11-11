
from __future__ import annotations

import json
import math
from itertools import combinations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import make_smoothing_spline
from scipy.signal import savgol_filter


x0 = sp.Symbol("x0")


@dataclass
class DiscoveryResult:
    equation: str
    source_equation: str
    method: str
    rmse_rate: float
    mae_rate: float
    r_squared_rate: float
    validation_trajectory_rmse: float | None
    validation_scope: str
    observations: int
    trajectories: int
    concentration_range: list[float]
    rate_units: str
    preprocessing: str
    candidates: list[dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _preprocess_group(group: pd.DataFrame, method: str) -> pd.DataFrame:
    group = group.sort_values("_time").copy()
    t = group["_time"].to_numpy(dtype=float)
    ca = group["_ca"].to_numpy(dtype=float)
    if len(t) < 7:
        raise ValueError("Each training trajectory needs at least 7 observations.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(ca)):
        raise ValueError("Time and concentration values must be finite.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time values must be unique and strictly increasing per trajectory.")

    if method == "Raw":
        group["CA_fit"] = ca
        group["rate_fit"] = -np.gradient(ca, t, edge_order=2)
        trim = 5
    elif method == "Savitzky-Golay":
        # Match the notebook's window/order when possible; shrink safely for shorter runs.
        window = min(11, len(group) if len(group) % 2 else len(group) - 1)
        polyorder = min(3, window - 2)
        dt = np.diff(t)
        if not np.allclose(dt, dt[0], rtol=1e-4, atol=1e-10):
            raise ValueError("Savitzky-Golay preprocessing requires equally spaced time points.")
        group["CA_fit"] = savgol_filter(ca, window_length=window, polyorder=polyorder)
        group["rate_fit"] = -savgol_filter(ca, window_length=window, polyorder=polyorder, deriv=1, delta=float(dt[0]))
        trim = window // 2
    elif method == "GCV spline":
        spline = make_smoothing_spline(t, ca, lam=None)
        group["CA_fit"] = np.asarray(spline(t), dtype=float)
        group["rate_fit"] = -np.asarray(spline.derivative()(t), dtype=float)
        trim = 5
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")
    trim = min(trim, max(1, (len(group) - 7) // 2))
    group["_valid"] = (np.arange(len(group)) >= trim) & (np.arange(len(group)) < len(group) - trim)
    return group


def _integrate_expression(expr: sp.Expr, time: np.ndarray, observed: np.ndarray) -> np.ndarray | None:
    """ODE reconstruction used to rank candidate laws on held-out trajectories."""
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        return None
    initial = max(float(observed[0]), 0.0)
    grid = np.linspace(max(initial * 1e-5, 1e-10), max(initial, 1e-10), 100)
    try:
        rates = _predict(expr, grid)
        if not np.all(np.isfinite(rates)) or np.any(rates < -1e-10):
            return None

        def rhs(_time: float, state: np.ndarray) -> list[float]:
            rate = float(_predict(expr, np.asarray(max(float(state[0]), 0.0))))
            if not np.isfinite(rate) or rate < -1e-8:
                raise ValueError("Nonphysical rate")
            return [-max(rate, 0.0)]

        solution = solve_ivp(rhs, (float(time[0]), float(time[-1])), [initial], t_eval=time, rtol=1e-7, atol=1e-10)
    except Exception:
        return None
    if not solution.success or solution.y.shape[1] != len(time) or not np.all(np.isfinite(solution.y[0])):
        return None
    return np.maximum(solution.y[0], 0.0)


def _candidate_terms(expr: sp.Expr) -> list[sp.Expr]:
    expanded = sp.expand(expr)
    return [term.as_coeff_Mul()[1] for term in sp.Add.make_args(expanded)]


def _simplification_candidates(expr: sp.Expr, max_terms: int = 5) -> list[sp.Expr]:
    """Build the bounded algebraic candidate set used by the analysis notebook.

    Keep this deliberately bounded: candidates are later ranked on untouched
    trajectory data, so the simplifier can explore useful reductions without
    selecting them from training fit alone.
    """
    candidates: dict[str, sp.Expr] = {}

    def add(candidate: sp.Expr) -> None:
        try:
            candidate = sp.cancel(candidate)
            if candidate not in (sp.nan, sp.zoo, sp.oo, -sp.oo):
                candidates[sp.srepr(candidate)] = candidate
        except (sp.SympifyError, ValueError, TypeError, ZeroDivisionError):
            return

    for form in (expr, sp.simplify(expr), sp.factor(expr), sp.cancel(expr), sp.together(expr), sp.expand(expr)):
        add(form)

    def subsets(terms: list[sp.Expr]) -> list[sp.Expr]:
        if not terms:
            return []
        if len(terms) > max_terms:
            return [sp.Add(*terms), *terms]
        return [sp.Add(*part) for size in range(1, len(terms) + 1) for part in combinations(terms, size)]

    expanded = sp.expand(expr)
    for candidate in subsets(list(sp.Add.make_args(expanded))):
        add(candidate)

    numerator, denominator = sp.fraction(sp.together(expr))
    numerator_terms = list(sp.Add.make_args(sp.expand(numerator)))
    denominator_terms = list(sp.Add.make_args(sp.expand(denominator)))
    numerator_forms = subsets(numerator_terms)
    denominator_forms = subsets(denominator_terms)
    for candidate in numerator_forms:
        add(candidate / denominator)
    for candidate in denominator_forms:
        if candidate != 0:
            add(numerator / candidate)
    if len(numerator_terms) <= 3 and len(denominator_terms) <= 3:
        for top in numerator_forms:
            for bottom in denominator_forms:
                if bottom != 0:
                    add(top / bottom)
    return list(candidates.values())


def _fit_coefficients(expr: sp.Expr, ca: np.ndarray, rate: np.ndarray) -> tuple[sp.Expr, float] | None:
    terms = _candidate_terms(expr)
    try:
        funcs = [sp.lambdify(x0, term, modules="numpy") for term in terms]
        matrix = np.column_stack([np.broadcast_to(fn(ca), ca.shape) for fn in funcs]).astype(float)
    except Exception:
        return None
    if not np.all(np.isfinite(matrix)):
        return None
    coeffs, *_ = np.linalg.lstsq(matrix, rate, rcond=None)
    pred = matrix @ coeffs
    if not np.all(np.isfinite(pred)):
        return None
    fitted = sp.simplify(sum(sp.Float(float(c), 12) * term for c, term in zip(coeffs, terms)))
    return fitted, float(np.mean((rate - pred) ** 2))


def _predict(expr: sp.Expr, ca: np.ndarray) -> np.ndarray:
    fn = sp.lambdify(x0, expr, modules="numpy")
    with np.errstate(all="ignore"):
        values = np.asarray(fn(ca), dtype=float)
    return np.broadcast_to(values, ca.shape)


def _complexity(expr: sp.Expr) -> int:
    return int(sp.count_ops(expr))


def discover_rate_law(
    data: pd.DataFrame,
    *,
    time_col: str = "time_min",
    ca_col: str = "CA",
    trajectory_col: str | None = None,
    niterations: int = 100,
    verbosity: int = 0,
) -> DiscoveryResult:
    """Discover a CA-only rate law, selecting preprocessing/equation by ODE holdout.

    When multiple trajectories are supplied, whole trajectories are held out.
    For a single trajectory, a final contiguous time segment is held out instead.
    """
    needed = [time_col, ca_col] + ([trajectory_col] if trajectory_col else [])
    missing = [name for name in needed if name not in data.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    frame = data[needed].copy().rename(columns={time_col: "_time", ca_col: "_ca"})
    if trajectory_col is None:
        frame["_trajectory"] = "trajectory_1"
    else:
        frame = frame.rename(columns={trajectory_col: "_trajectory"})
    frame = frame.dropna(subset=["_time", "_ca", "_trajectory"])
    if (frame["_ca"] < 0).any():
        raise ValueError("Concentrations must be nonnegative.")
    raw_groups = [group.sort_values("_time").copy() for _, group in frame.groupby("_trajectory", sort=False)]
    for group in raw_groups:
        t = group["_time"].to_numpy(float)
        ca = group["_ca"].to_numpy(float)
        if len(group) < 12:
            raise ValueError("Each trajectory needs at least 12 observations for a stable train/validation fit.")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(ca)) or np.any(np.diff(t) <= 0):
            raise ValueError("Each trajectory needs finite data and unique, strictly increasing time values.")
        if np.any(ca < 0):
            raise ValueError("Concentrations must be nonnegative.")

    validation_scope: str
    train_groups: list[pd.DataFrame]
    validation_groups: list[pd.DataFrame]
    if len(raw_groups) == 1:
        # Keep the last fifth untouched until model selection. This tests short-range
        # forward prediction instead of rewarding an in-sample derivative fit.
        group = raw_groups[0]
        cut = max(10, int(len(group) * 0.8))
        cut = min(cut, len(group) - 5)
        train_groups = [group.iloc[:cut].copy()]
        validation_groups = [group.iloc[cut:].copy()]
        validation_scope = "last 20% of the only trajectory (contiguous time holdout)"
    else:
        # Deterministic trajectory holdout prevents rows from the same run leaking
        # across training and validation.
        rng = np.random.default_rng(657)
        order = rng.permutation(len(raw_groups))
        n_validation = max(1, int(math.ceil(0.2 * len(raw_groups))))
        n_validation = min(n_validation, len(raw_groups) - 1)
        validation_indices = set(order[:n_validation].tolist())
        train_groups = [g for i, g in enumerate(raw_groups) if i not in validation_indices]
        validation_groups = [g for i, g in enumerate(raw_groups) if i in validation_indices]
        validation_scope = f"{len(validation_groups)} held-out trajectory/trajectories"

    warnings: list[str] = []
    if len(raw_groups) == 1:
        warnings.append("Only one trajectory supplied; validation is a time-block holdout, not an independent run.")

    # PySR is imported lazily so upload parsing and UI startup do not require Julia initialization.
    try:
        from pysr import PySRRegressor
    except ImportError as exc:
        raise RuntimeError("PySR is required. Install dependencies from requirements.txt.") from exc
    except Exception as exc:
        raise RuntimeError(
            "PySR could not initialize its Julia backend. Close any other process initializing PySR/Julia, "
            "then retry; the backend may also need its first-time setup to finish."
        ) from exc
    methods = ["Raw", "Savitzky-Golay", "GCV spline"]
    validation_rows: list[dict[str, Any]] = []
    training_context: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method_name in methods:
        try:
            prepared = pd.concat([_preprocess_group(g, method_name) for g in train_groups], ignore_index=True)
        except ValueError as exc:
            warnings.append(f"Skipped {method_name}: {exc}")
            continue
        usable = prepared.loc[
            prepared["_valid"] & (prepared["CA_fit"] > 1e-10)
            & np.isfinite(prepared["rate_fit"])
        ].copy()
        if len(usable) < 7:
            continue
        ca_train = usable["CA_fit"].to_numpy(float)
        rate_train = usable["rate_fit"].to_numpy(float)
        if np.ptp(ca_train) <= max(abs(float(np.mean(ca_train))), 1.0) * 1e-8:
            continue
        training_context[method_name] = (ca_train, rate_train)
        model = PySRRegressor(
            niterations=niterations,
            populations=15,
            population_size=33,
            maxsize=20,
            binary_operators=["+", "-", "*", "/", "^"],
            constraints={"^": (-1, 1)},
            model_selection="best",
            verbosity=verbosity,
            progress=False,
        )
        model.fit(ca_train.reshape(-1, 1), rate_train)
        seen: set[str] = set()
        for _, row in model.equations_.sort_values(["loss", "complexity"]).iterrows():
            source = sp.sympify(row["sympy_format"])
            for structure in _simplification_candidates(source):
                fit = _fit_coefficients(structure, ca_train, rate_train)
                if fit is None:
                    continue
                expression, train_mse = fit
                key = str(expression)
                if key in seen:
                    continue
                seen.add(key)
                grid = np.linspace(1e-10, max(float(np.max(ca_train)), 1e-10), 120)
                physical_rates = _predict(expression, grid)
                if not np.all(np.isfinite(physical_rates)) or np.mean(physical_rates < -1e-10) > 0.02:
                    continue
                errors: list[float] = []
                for validation in validation_groups:
                    validation = validation.sort_values("_time")
                    time = validation["_time"].to_numpy(float)
                    observed = validation["_ca"].to_numpy(float)
                    # For single-run time holdout, initialize from the measured CA at
                    # the block boundary; for held-out trajectories use its first point.
                    pred_ca = _integrate_expression(expression, time, observed)
                    if pred_ca is None:
                        errors = []
                        break
                    errors.append(float(np.sqrt(np.mean((observed - pred_ca) ** 2))))
                if not errors:
                    continue
                validation_rmse = float(np.mean(errors))
                validation_rows.append({
                    "equation": key,
                    "source_equation": str(source),
                    "method": method_name,
                    "rmse_rate": float(np.sqrt(train_mse)),
                    "complexity": _complexity(expression),
                    "train_loss": float(row["loss"]),
                    "validation_trajectory_rmse": validation_rmse,
                    "validation_se": float(np.std(errors, ddof=1) / np.sqrt(len(errors))) if len(errors) > 1 else 0.0,
                })
    if not validation_rows:
        raise RuntimeError("No candidate equations could be integrated across the validation data. Check concentration range and time units.")

    # Apply the notebook's one-standard-error rule within each method, then
    # compare the compact method winners under a global one-SE rule. No method
    # gets a hard-coded preference: the validation trajectories decide.
    method_winners: list[dict[str, Any]] = []
    for method_name in methods:
        candidates = [r for r in validation_rows if r["method"] == method_name]
        if not candidates:
            continue
        best_validation = min(candidates, key=lambda r: r["validation_trajectory_rmse"])
        threshold = best_validation["validation_trajectory_rmse"] + best_validation["validation_se"]
        eligible = [r for r in candidates if r["validation_trajectory_rmse"] <= threshold]
        method_winners.append(min(eligible, key=lambda r: (r["complexity"], r["validation_trajectory_rmse"], r["train_loss"])))
    best_validation = min(method_winners, key=lambda r: r["validation_trajectory_rmse"])
    global_threshold = best_validation["validation_trajectory_rmse"] + best_validation["validation_se"]
    globally_eligible = [r for r in method_winners if r["validation_trajectory_rmse"] <= global_threshold]
    best = min(globally_eligible, key=lambda r: (r["complexity"], r["validation_trajectory_rmse"]))
    rows = sorted(validation_rows, key=lambda r: (r["validation_trajectory_rmse"], r["complexity"], r["rmse_rate"]))
    equation = sp.sympify(best["equation"])
    ca_train, rate_train = training_context[best["method"]]
    pred = _predict(equation, ca_train)
    residual = rate_train - pred
    ss_tot = float(np.sum((rate_train - np.mean(rate_train)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / ss_tot if ss_tot > 0 else float("nan")
    return DiscoveryResult(
        equation=best["equation"],
        source_equation=best["source_equation"],
        method=best["method"],
        rmse_rate=float(np.sqrt(np.mean(residual**2))),
        mae_rate=float(np.mean(np.abs(residual))),
        r_squared_rate=r2,
        validation_trajectory_rmse=best["validation_trajectory_rmse"],
        validation_scope=validation_scope,
        observations=int(len(ca_train)),
        trajectories=len(raw_groups),
        concentration_range=[float(ca_train.min()), float(ca_train.max())],
        rate_units="concentration unit / time unit",
        preprocessing=f"Selected {best['method']}; candidates simplified/coefficient-refit on training rates, ranked by held-out ODE trajectory RMSE with a one-standard-error rule",
        candidates=rows[:10],
        warnings=warnings,
    )


def predict_rate(equation: str, concentration: float | list[float] | np.ndarray) -> float | np.ndarray:
    """Evaluate an exported equation at nonnegative concentration values."""
    values = np.asarray(concentration, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Concentration values must be finite and nonnegative.")
    prediction = _predict(sp.sympify(equation), values)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Equation is undefined at the supplied concentration.")
    return float(prediction) if prediction.ndim == 0 else prediction


def integrate_rate_law(
    equation: str, times: list[float] | np.ndarray, initial_ca: float
) -> np.ndarray:
    """Integrate dCA/dt = -rate(CA) over the requested increasing times."""
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) < 2 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Times must be a finite, strictly increasing vector with at least two values.")
    if initial_ca < 0 or not np.isfinite(initial_ca):
        raise ValueError("Initial concentration must be finite and nonnegative.")
    expr = sp.sympify(equation)

    def rhs(_t: float, state: np.ndarray) -> list[float]:
        ca = max(float(state[0]), 0.0)
        rate = float(_predict(expr, np.asarray(ca)))
        if not np.isfinite(rate):
            raise ValueError("Rate equation produced a non-finite value.")
        return [-max(rate, 0.0)]

    solution = solve_ivp(rhs, (float(t[0]), float(t[-1])), [float(initial_ca)], t_eval=t, rtol=1e-7, atol=1e-10)
    if not solution.success:
        raise RuntimeError(solution.message)
    return np.maximum(solution.y[0], 0.0)


def export_result(result: DiscoveryResult, path: str | Path) -> Path:
    """Write a portable JSON model card containing the simplified equation and metrics."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return destination
