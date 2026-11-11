"""PID Dataset v1.1 — corrected Pilot only (18 sequences, 8192 samples).

Requirements: Python 3.10+, numpy, matplotlib.
Usage: python pid_dataset_pilot.py --q 4 --seed 20261102 --out pid_pilot
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TS = 10.0
SIGNALS = (
    "step", "ramp_triangle", "multi_sine", "prbs_pulses",
    "noise", "accumulate_hold", "sign_compensate", "combined",
)
WEIGHTS_Q4 = np.array([
    [0.4, 0.3, -0.2, 0.1],
    [-0.1, 0.4, 0.3, -0.2],
    [0.1, -0.3, 0.2, 0.4],
], dtype=np.float64)
NOMINAL = np.array([1.0, 0.005, 5.0], dtype=np.float64)
REL_SPAN = np.array([0.3, 0.4, 0.4], dtype=np.float64)
LIMITS = np.array([[0.7, 1.3], [0.003, 0.007], [3.0, 7.0]], dtype=np.float64)
CHECK_ATOL = 1e-10
CHECK_RTOL = 1e-11


def check_weight_vectors(weights: np.ndarray, q: int) -> dict:
    """Check shape/L1 normalization; nonproportionality only when q >= 2."""
    if weights.shape != (3, q):
        raise AssertionError(f"Expected coefficient weight shape (3,{q})")
    if not np.all(np.isfinite(weights)):
        raise AssertionError("Non-finite coefficient weights")
    norms = np.sum(np.abs(weights), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-12, rtol=0):
        raise AssertionError("Each weight vector must have L1 norm 1")
    if q == 1:
        # Every pair of nonzero 1D vectors is necessarily proportional.
        return {"status": "PASS", "q": 1, "nonproportionality": "NOT_APPLICABLE",
                "note": "All nonzero one-dimensional vectors are proportional; this is not an error.",
                "l1_norms": norms.tolist(), "pairwise_cosines": None}
    names = ("Kp/Ki", "Kp/Kd", "Ki/Kd")
    pairs = ((0, 1), (0, 2), (1, 2))
    cosines = {}
    for name, (i, j) in zip(names, pairs):
        cosine = float(abs(np.dot(weights[i], weights[j])) /
                       (np.linalg.norm(weights[i]) * np.linalg.norm(weights[j])))
        cosines[name] = cosine
        if np.linalg.matrix_rank(weights[[i, j]], tol=1e-10) < 2 or cosine >= 0.95:
            raise AssertionError(f"Coefficient weight vectors are proportional or too close (|cos| >= 0.95): {name}")
    return {"status": "PASS", "q": q, "nonproportionality": "PASS",
            "l1_norms": norms.tolist(), "pairwise_abs_cosines": cosines}


def coefficient_weights(q: int, seed: int) -> np.ndarray:
    if q < 1:
        raise ValueError("q must be >= 1")
    if q == 4:
        result = WEIGHTS_Q4.copy()
    else:
        rng = np.random.default_rng(seed)
        # q=1 is valid: three independent gain mappings are still definable,
        # even though their 1D weight vectors cannot be nonproportional.
        for _ in range(1000):
            result = rng.normal(size=(3, q))
            row_norms = np.sum(np.abs(result), axis=1, keepdims=True)
            if np.any(row_norms == 0):
                continue
            result = (result / row_norms).astype(np.float64)
            try:
                check_weight_vectors(result, q)
                break
            except AssertionError:
                continue
        else:
            raise RuntimeError("Unable to generate valid independent coefficient weight vectors")
    check_weight_vectors(result, q)
    return result


def coefficients(z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return Kp, Ki, Kd for z with shape (n, q), no clipping."""
    return NOMINAL * (1.0 + REL_SPAN * (z @ weights.T))


def rescale_signal(values: np.ndarray, amplitude: float) -> np.ndarray:
    """Scale complete *input signal* without clipping or changing PID law."""
    peak = float(np.max(np.abs(values)))
    if peak == 0:
        return values.copy()
    return (values * (amplitude / peak)).astype(np.float64)


def balanced_pulses(n: int, rng: np.random.Generator,
                    short_only: bool = False) -> np.ndarray:
    """Alternating pulses with equal signed area, plus exact-zero intervals."""
    x = np.zeros(n, dtype=np.float64)
    pos = 0
    while pos < n:
        dwell = int(rng.integers(1, 4) if short_only or rng.random() < 0.82
                    else rng.integers(8, 31))
        gap = int(rng.integers(1, 3))
        cycle = dwell * 2 + gap * 2
        if pos + cycle > n:
            break
        a = float(rng.uniform(0.3, 1.0) * rng.choice([-1.0, 1.0]))
        x[pos:pos+dwell] = a
        x[pos+dwell+gap:pos+2*dwell+gap] = -a
        pos += cycle
    return x


def make_error(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    amplitude = float(rng.uniform(0.18, 1.0))
    x = np.zeros(n, dtype=np.float64)

    if kind == "step":
        pos = int(rng.integers(8, 25))
        while pos < n:
            level = float(rng.uniform(0.1, 1.0)) * rng.choice([-1.0, 1.0])
            length = int(rng.integers(12, 75))
            x[pos:min(n, pos + length)] = level
            pos += length
    elif kind == "ramp_triangle":
        period = int(rng.integers(25, min(120, n) + 1))
        phase = float(rng.uniform(-np.pi, np.pi))
        x = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * t / period + phase))
    elif kind == "multi_sine":
        p1 = int(rng.integers(30, min(150, n) + 1))
        p2 = int(rng.integers(4, 25))
        x = 0.75 * np.sin(2 * np.pi * t / p1 + rng.uniform(0, 2*np.pi))
        x += 0.25 * np.sin(2 * np.pi * t / p2 + rng.uniform(0, 2*np.pi))
    elif kind == "prbs_pulses":
        # Balanced bipolar pulse pairs: includes rapid changes, zero intervals,
        # and occasional longer plateaus without long-term integral drift.
        x = balanced_pulses(n, rng)
    elif kind == "noise":
        white = rng.normal(size=n)
        correlated = np.empty(n, dtype=np.float64)
        state = 0.0
        for k in range(n):
            state = 0.86*state + 0.14*white[k]
            correlated[k] = state
        # High-pass-like variation, centered to avoid inadvertent I dominance.
        x = 0.3*correlated + 0.7*(white - np.convolve(white, np.ones(7)/7, mode="same"))
        x -= x.mean()
    elif kind == "accumulate_hold":
        x[:max(1, n//3)] = float(rng.uniform(0.15, 0.45)) * rng.choice([-1.0, 1.0])
        return x  # exact zero hold interval; within [-1, 1]
    elif kind == "sign_compensate":
        m = max(1, n//3)
        a = float(rng.uniform(0.12, 0.55)) * rng.choice([-1.0, 1.0])
        x[:m] = a
        x[m:2*m] = -a
        return x  # equal positive/negative areas, then exact zeros
    elif kind == "combined":
        cuts = np.linspace(0, n, 6, dtype=int)
        x[cuts[0]:cuts[1]] = balanced_pulses(cuts[1]-cuts[0], rng, True)
        s = np.arange(cuts[2] - cuts[1])
        x[cuts[1]:cuts[2]] = 0.65 - 1.3 * s / max(1, len(s)-1)
        s = np.arange(cuts[3] - cuts[2])
        x[cuts[2]:cuts[3]] = 0.5*np.sin(2*np.pi*s/9)
        x[cuts[3]:cuts[4]] = balanced_pulses(cuts[4]-cuts[3], rng, True)
        x[cuts[4]:cuts[5]] = 0.0
    elif kind == "long_hold":
        x[:n//8] = 0.18
        return x
    elif kind == "long_z_comp":
        m = max(1, n//10)
        x[:m] = 0.2
        start_neg = n//2
        x[start_neg:start_neg+m] = -0.2
        # After integral compensation, add balanced disturbances and finish
        # with a zero-error hold; integral is never forcibly reset.
        start_pulses, end_pulses = int(0.66*n), int(0.84*n)
        x[start_pulses:end_pulses] = 0.7 * balanced_pulses(end_pulses-start_pulses, rng, True)
        return x
    else:
        raise ValueError(f"Unknown error signal: {kind}")
    return rescale_signal(x, amplitude)


def make_z(mode: str, n: int, q: int, rng: np.random.Generator,
           kind: str, weights: np.ndarray) -> np.ndarray:
    if kind == "long_z_comp":
        # Keep integrator unchanged while Ki moves markedly during e == 0.
        direction = np.where(weights[1] >= 0, 1.0, -1.0)
        low, high = -0.7*direction, 0.7*direction
        z = np.empty((n, q), dtype=np.float64)
        left, right = n//5, n//3
        z[:left] = low
        blend = np.linspace(0.0, 1.0, right-left, dtype=np.float64)[:, None]
        z[left:right] = low + blend*(high-low)
        z[right:] = high
        return z

    if mode == "constant":
        return np.broadcast_to(rng.uniform(-0.85, 0.85, size=q), (n, q)).copy()
    if mode == "smooth":
        z = np.empty((n, q), dtype=np.float64)
        for j in range(q):
            times = [0]
            while times[-1] < n - 1:
                times.append(times[-1] + int(rng.integers(180, 1081)))
            anchors = rng.uniform(-0.9, 0.9, size=len(times))
            z[:, j] = np.interp(np.arange(n), times, anchors)
        return z
    if mode == "step":
        # Old implementation drew the FIRST dwell from [180, 1080], so a
        # 256-step sequence sometimes contained no transition at all.
        # For the short Pilot only, force the first switch to occur inside
        # [20%, 60%] of the trajectory; subsequent dwells remain slow.
        z = np.empty((n, q), dtype=np.float64)
        first_lo = max(1, int(np.ceil(0.20*n)))
        first_hi = min(n-1, int(np.floor(0.60*n)))
        if first_hi < first_lo:
            raise ValueError("Step trajectory needs at least two samples")
        first_switch = int(rng.integers(first_lo, first_hi+1))
        current = rng.uniform(-0.9, 0.9, size=q)
        z[:first_switch] = current
        pos = first_switch
        while pos < n:
            # Enforce an *actual* vector change, not merely a new timestamp.
            for _ in range(100):
                next_level = rng.uniform(-0.9, 0.9, size=q)
                if float(np.max(np.abs(next_level-current))) >= 0.2:
                    break
            else:
                raise AssertionError("Failed to create a genuine z step")
            length = int(rng.integers(180, 1081))
            z[pos:min(n, pos+length)] = next_level
            current = next_level
            pos += length
        return z
    raise ValueError(f"Unknown z mode: {mode}")


def reference_pid(error: np.ndarray, gains: np.ndarray) -> dict[str, np.ndarray]:
    """One independent stateful PID run, float64, no clipping/anti-windup."""
    n = error.size
    p = np.empty(n, dtype=np.float64)
    integ = np.empty(n, dtype=np.float64)
    deriv = np.empty(n, dtype=np.float64)
    prev_e = np.float64(0.0)
    accum = np.float64(0.0)
    for k in range(n):
        ek = np.float64(error[k])
        p[k] = ek
        accum = np.float64(accum + TS*ek)
        integ[k] = accum
        deriv[k] = np.float64((ek-prev_e)/TS)
        prev_e = ek
    up = gains[:, 0] * p
    ui = gains[:, 1] * integ
    ud = gains[:, 2] * deriv
    return {"P": p, "I": integ, "D": deriv,
            "u_P": up, "u_I": ui, "u_D": ud, "u": up+ui+ud}


def build_specs(seed: int) -> list[dict]:
    """Exactly 16 short (two of each type) and two distinct long sequences."""
    specs = []
    for kind in SIGNALS:
        for repeat in ("A", "B"):
            specs.append({"sequence_id": f"{kind}_{repeat}", "signal_type": kind,
                          "length": 256})
    specs += [
        {"sequence_id": "long_hold", "signal_type": "long_hold", "length": 2048,
         "z_mode": "constant"},
        {"sequence_id": "long_z_comp", "signal_type": "long_z_comp", "length": 2048,
         "z_mode": "smooth"},
    ]
    rng = np.random.default_rng(seed)
    modes = ["constant"]*3 + ["smooth"]*10 + ["step"]*3
    rng.shuffle(modes)
    for item, mode in zip(specs[:16], modes):
        item["z_mode"] = mode
    for item in specs:
        item["error_seed"] = int(rng.integers(0, 2**32, dtype=np.uint32))
        item["z_seed"] = int(rng.integers(0, 2**32, dtype=np.uint32))
    return specs


def make_sequence(spec: dict, q: int, weights: np.ndarray) -> dict[str, np.ndarray]:
    n = spec["length"]
    e = make_error(spec["signal_type"], n, np.random.default_rng(spec["error_seed"]))
    z = make_z(spec["z_mode"], n, q, np.random.default_rng(spec["z_seed"]),
               spec["signal_type"], weights)
    gains = coefficients(z, weights)
    return {"e": e, "z": z, "gains": gains, **reference_pid(e, gains)}


def columns(q: int) -> list[str]:
    return (["sequence_id", "k", "time_s", "signal_type", "z_mode", "e"]
            + [f"z_{j+1}" for j in range(q)]
            + ["Kp", "Ki", "Kd", "P", "I", "D", "u_P", "u_I", "u_D", "u"])


def write_data(path: Path, specs: list[dict], sequences: dict, q: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns(q))
        for spec in specs:
            dat = sequences[spec["sequence_id"]]
            for k in range(spec["length"]):
                numbers = ([dat["e"][k]] + list(dat["z"][k])
                           + list(dat["gains"][k])
                           + [dat[name][k] for name in ("P", "I", "D", "u_P", "u_I", "u_D", "u")])
                writer.writerow([spec["sequence_id"], k, int(k*TS),
                                 spec["signal_type"], spec["z_mode"]]
                                + [format(float(v), ".17g") for v in numbers])


def write_sequence_index(path: Path, specs: list[dict]) -> None:
    fields = ["sequence_id", "signal_type", "z_mode", "length", "error_seed", "z_seed"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(specs)


def check_data(csv_path: Path, specs: list[dict], weights: np.ndarray,
               q: int) -> dict:
    """Independent CSV re-read; check every numeric identity and initial state."""
    specs_by_id = {s["sequence_id"]: s for s in specs}
    counts = Counter()
    previous_e = {sid: 0.0 for sid in specs_by_id}
    previous_i = {sid: 0.0 for sid in specs_by_id}
    last_k = {sid: -1 for sid in specs_by_id}
    maximum_residual = 0.0
    component_dominance = Counter()
    informative = 0
    zero_hold_checks = 0
    max_abs_u = 0.0

    def insist(actual: float, expected: float, what: str, sid: str, k: int) -> None:
        nonlocal maximum_residual
        residual = abs(actual-expected)
        maximum_residual = max(maximum_residual, residual)
        if not np.isclose(actual, expected, rtol=CHECK_RTOL, atol=CHECK_ATOL):
            raise AssertionError(f"{what} mismatch at {sid}, k={k}: {actual} != {expected}")

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != columns(q):
            raise AssertionError("Wrong CSV header")
        for row in reader:
            sid = row["sequence_id"]
            if sid not in specs_by_id:
                raise AssertionError(f"Unknown sequence: {sid}")
            spec = specs_by_id[sid]
            k = int(row["k"])
            if k != last_k[sid] + 1:
                raise AssertionError(f"Out-of-order steps for {sid}")
            last_k[sid] = k
            counts[sid] += 1
            if int(row["time_s"]) != int(k*TS):
                raise AssertionError(f"Wrong time_s for {sid}, k={k}")
            if row["signal_type"] != spec["signal_type"] or row["z_mode"] != spec["z_mode"]:
                raise AssertionError("Metadata mismatch")
            nums = {name: float(row[name]) for name in columns(q)[5:]}
            if not all(np.isfinite(list(nums.values()))):
                raise AssertionError(f"Nonfinite data at {sid}, k={k}")
            e = nums["e"]
            z = np.array([nums[f"z_{j+1}"] for j in range(q)], dtype=np.float64)
            if abs(e) > 1+1e-12 or np.any(np.abs(z) > 1+1e-12):
                raise AssertionError("Input out of [-1, 1]")
            expected_k = coefficients(z[None, :], weights)[0]
            for name, actual, expected, (low, high) in zip(
                    ("Kp", "Ki", "Kd"),
                    (nums["Kp"], nums["Ki"], nums["Kd"]), expected_k, LIMITS):
                insist(actual, float(expected), name, sid, k)
                if not low-1e-12 <= actual <= high+1e-12:
                    raise AssertionError(f"{name} out of bounds")
            iprev, eprev = previous_i[sid], previous_e[sid]
            insist(nums["P"], e, "P", sid, k)
            insist(nums["I"], iprev+TS*e, "I", sid, k)
            insist(nums["D"], (e-eprev)/TS, "D", sid, k)
            for part, gain, signal in (("u_P", "Kp", "P"),
                                       ("u_I", "Ki", "I"),
                                       ("u_D", "Kd", "D")):
                insist(nums[part], nums[gain]*nums[signal], part, sid, k)
            insist(nums["u"], nums["u_P"]+nums["u_I"]+nums["u_D"], "u", sid, k)
            if k>0 and e==0 and eprev==0:
                insist(nums["I"], iprev, "zero-error hold", sid, k)
                zero_hold_checks += 1
            previous_i[sid] = nums["I"]
            previous_e[sid] = e
            max_abs_u = max(max_abs_u, abs(nums["u"]))
            parts = np.abs([nums["u_P"], nums["u_I"], nums["u_D"]])
            if np.sum(parts) > 1e-8:
                informative += 1
                who = int(np.argmax(parts))
                if parts[who]/np.sum(parts) > 0.5:
                    component_dominance[("P", "I", "D")[who]] += 1
    if any(counts[s["sequence_id"]] != s["length"] for s in specs):
        raise AssertionError("Wrong samples per sequence")
    if len(specs) != 18 or sum(counts.values()) != 8192:
        raise AssertionError("Pilot must have exactly 18 sequences and 8192 rows")
    # Z-mode distribution, zero-error holds and observed |u| > 1 are
    # coverage diagnostics; arithmetic correctness is checked above.
    return {"status": "PASS", "rows_checked": sum(counts.values()),
            "sequences_checked": len(counts), "max_absolute_equation_residual": maximum_residual,
            "zero_error_memory_checks": zero_hold_checks,
            "max_absolute_u": max_abs_u,
            "dominance_fraction_of_informative": {
                key: round(component_dominance[key]/informative, 6)
                for key in ("P", "I", "D")},
            "dominance_goal_at_least_5_percent": {
                key: component_dominance[key]/informative >= 0.05
                for key in ("P", "I", "D")},
            "dominance_note": "Coverage goals are diagnostics, not PID constraints.",
            "checks": ["CSV schema", "row counts", "time order", "finite float64 values",
                       "e/z ranges", "K ranges", "linear K(z)", "P=error",
                       "stateful integral", "backward difference", "each PID contribution",
                       "sum of contributions", "zero-error hold", "no output clipping"]}



def check_coverage(csv_path: Path, specs: list[dict], weights: np.ndarray,
                   q: int) -> dict:
    """Independent coverage audit of the actual saved CSV, not a PID test."""
    rows_by_id = {spec["sequence_id"]: [] for spec in specs}
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["sequence_id"] not in rows_by_id:
                raise AssertionError("Unknown sequence_id while auditing coverage")
            rows_by_id[row["sequence_id"]].append(row)

    checks = {}
    def record(name: str, ok: bool, detail) -> None:
        checks[name] = {"status": "PASS" if ok else "FAIL", "detail": detail}

    short = [s for s in specs if s["length"] == 256]
    long = [s for s in specs if s["length"] == 2048]
    families = dict(Counter(s["signal_type"] for s in short))
    record("all_e_signal_families", len(short) == 16 and all(families.get(x) == 2 for x in SIGNALS),
           {"counts_short": families, "expected_per_family": 2})
    record("long_memory_cases", len(long) == 2 and
           {s["sequence_id"] for s in long} == {"long_hold", "long_z_comp"},
           {"sequence_ids": [s["sequence_id"] for s in long]})
    record("sample_counts", len(specs) == 18 and sum(len(v) for v in rows_by_id.values()) == 8192 and
           all(len(rows_by_id[s["sequence_id"]]) == s["length"] for s in specs),
           {"sequences": len(specs), "rows": sum(map(len, rows_by_id.values()))})

    modes = dict(Counter(s["z_mode"] for s in specs))
    record("z_mode_counts", modes == {"constant": 4, "smooth": 11, "step": 3},
           {"actual": modes, "expected": {"constant": 4, "smooth": 11, "step": 3}})

    step_counts = {}
    const_max_change = {}
    smooth_counts = {}
    e_range_by_family = {}
    for spec in specs:
        sid = spec["sequence_id"]
        rows = rows_by_id[sid]
        z = np.asarray([[float(row[f"z_{j+1}"]) for j in range(q)] for row in rows])
        e = np.asarray([float(row["e"]) for row in rows])
        variation = np.max(np.abs(np.diff(z, axis=0)), axis=1)
        transitions = int(np.count_nonzero(variation > 1e-10))
        if spec["z_mode"] == "step":
            step_counts[sid] = transitions
        elif spec["z_mode"] == "constant":
            const_max_change[sid] = float(np.max(variation))
        elif spec["z_mode"] == "smooth":
            smooth_counts[sid] = transitions
        if spec["length"] == 256:
            e_range_by_family[sid] = {"min": float(e.min()), "max": float(e.max()),
                                       "amplitude_span": float(np.ptp(e))}
    record("step_z_actual_transitions", len(step_counts) == 3 and all(x >= 1 for x in step_counts.values()),
           {"step_sequence_transitions": step_counts, "minimum_per_sequence": 1})
    record("constant_z_is_constant", all(x <= 1e-12 for x in const_max_change.values()),
           {"max_consecutive_change": const_max_change})
    record("smooth_z_not_degenerate", all(x > 0 for x in smooth_counts.values()),
           {"sample_to_sample_changes": smooth_counts})
    record("error_families_not_degenerate", len(e_range_by_family) == 16 and
           all(x["amplitude_span"] > 0.05 for x in e_range_by_family.values()),
           e_range_by_family)

    coefficient_observed = {}
    for idx, name in enumerate(("Kp", "Ki", "Kd")):
        actual = np.asarray([float(row[name]) for rows in rows_by_id.values() for row in rows])
        low, high = LIMITS[idx]
        observed_low = float(np.min(actual))
        observed_high = float(np.max(actual))
        span_ratio = float((observed_high-observed_low)/(high-low))
        coefficient_observed[name] = {"observed_min": observed_low,
                                      "observed_max": observed_high,
                                      "theoretical_min": float(low),
                                      "theoretical_max": float(high),
                                      "fraction_of_theoretical_span": span_ratio}
    record("coefficient_actual_span", all(v["fraction_of_theoretical_span"] >= 0.20
                                          for v in coefficient_observed.values()),
           {"minimum_fraction": 0.20, "coefficients": coefficient_observed,
            "note": "No requirement to hit theoretical endpoints in random Pilot trajectories."})

    try:
        weight_audit = check_weight_vectors(weights, q)
    except AssertionError as exc:
        weight_audit = {"status": "FAIL", "error": str(exc)}
    record("coefficient_weight_vectors", weight_audit["status"] == "PASS", weight_audit)

    dominance = {}
    activity = {}
    for group, selected in (("short", short), ("long", long), ("all", specs)):
        sample_rows = [r for s in selected for r in rows_by_id[s["sequence_id"]]]
        raw = np.asarray([[float(r[x]) for x in ("u_P", "u_I", "u_D")]
                          for r in sample_rows], dtype=np.float64)
        magnitudes = np.abs(raw)
        total = np.sum(magnitudes, axis=1)
        mask = total > 1e-8
        ratios = np.zeros_like(magnitudes)
        ratios[mask] = magnitudes[mask]/total[mask, None]
        active = int(np.count_nonzero(mask))
        domination_counts = {}
        domination_percent = {}
        nonzero_counts = {}
        for idx, part in enumerate(("P", "I", "D")):
            count = int(np.count_nonzero(mask & (np.argmax(magnitudes, axis=1) == idx) &
                                          (ratios[:, idx] > 0.5)))
            domination_counts[part] = count
            domination_percent[part] = float(100*count/active) if active else 0.0
            nonzero_counts[part] = int(np.count_nonzero(magnitudes[:, idx] > 1e-8))
        dominance[group] = {"rows": len(sample_rows), "informative_rows": active,
                             "dominating_rows": domination_counts,
                             "dominating_percent": domination_percent}
        activity[group] = {"nonzero_rows_by_component": nonzero_counts}
    record("short_PID_dominance_at_least_5pct",
           all(x >= 5.0 for x in dominance["short"]["dominating_percent"].values()),
           {"percentages": dominance["short"]["dominating_percent"], "threshold_pct": 5.0})
    # The long sequences intentionally emphasize the integral. Reporting
    # overall/long dominance is useful; imposing a 5% threshold on them would
    # incorrectly penalize intended integrator-hold behavior.
    record("long_PID_nonzero_activity",
           all(x > 0 for x in activity["long"]["nonzero_rows_by_component"].values()),
           {"nonzero_rows": activity["long"]["nonzero_rows_by_component"],
            "note": "Long-set 5% dominance is diagnostic only; integral is expected to dominate."})

    all_rows = [r for rows in rows_by_id.values() for r in rows]
    u_all = np.asarray([float(r["u"]) for r in all_rows])
    e_zero_hold = 0
    for spec in specs:
        sid = spec["sequence_id"]
        e_arr = np.asarray([float(r["e"]) for r in rows_by_id[sid]])
        e_zero_hold += int(np.count_nonzero((e_arr[1:] == 0) & (e_arr[:-1] == 0)))
    record("zero_error_hold_cases_present", e_zero_hold > 0,
           {"consecutive_zero_error_pairs": e_zero_hold})
    record("unclipped_output_evidence", float(np.max(np.abs(u_all))) > 1.0,
           {"max_absolute_u": float(np.max(np.abs(u_all))),
            "note": "Abs(u)>1 is useful coverage evidence; mathematical sum check rules out clipping."})

    e_all = np.array([float(r["e"]) for rows in rows_by_id.values() for r in rows])
    z_all = np.array([[float(r[f"z_{j+1}"]) for j in range(q)]
                      for rows in rows_by_id.values() for r in rows])
    observed_inputs = {"e": {"min": float(e_all.min()), "max": float(e_all.max())},
                       "z_each": [{"min": float(z_all[:, j].min()),
                                    "max": float(z_all[:, j].max())}
                                   for j in range(q)],
                       "z_overall": {"min": float(z_all.min()),
                                     "max": float(z_all.max())}}
    return {"status": "PASS" if all(x["status"] == "PASS" for x in checks.values()) else "FAIL",
            "checks": checks, "dominance_by_duration": dominance,
            "component_activity_by_duration": activity,
            "observed_input_ranges": observed_inputs,
            "observed_coefficient_ranges": coefficient_observed,
            "step_transition_counts": step_counts,
            "notes": ["Pilot normalized input coverage does not need to hit +/-1.",
                      "Theoretical K bounds are guaranteed by L1-normalized weights, not by clipping.",
                      "Input signal amplitude is adjusted before the reference PID; its states/output are never clipped.",
                      "Long-run integral accumulation can dominate P/D; report long dominance separately."]}


def unit_tests(weights: np.ndarray, q: int, sequences: dict) -> dict:
    """Extra independent tests for known numeric cases and long memory."""
    nominal = coefficients(np.zeros((1, q), dtype=np.float64), weights)[0]
    np.testing.assert_allclose(nominal, NOMINAL, rtol=0, atol=1e-14)
    if q == 4:
        z_example = np.array([[0.5, -0.5, 0.0, 1.0]], dtype=np.float64)
        np.testing.assert_allclose(coefficients(z_example, weights)[0],
                                   [1.045, 0.0041, 6.2], rtol=0, atol=1e-14)
        corners = np.array(np.meshgrid(*[[-1.0, 1.0]]*4)).T.reshape(-1, 4)
        corner_gains = coefficients(corners, weights)
        np.testing.assert_allclose(corner_gains.min(axis=0), LIMITS[:, 0], atol=1e-14)
        np.testing.assert_allclose(corner_gains.max(axis=0), LIMITS[:, 1], atol=1e-14)

    e = np.array([0, 0.4, 0.4, 0, 0], dtype=np.float64)
    gains = np.broadcast_to(nominal, (len(e), 3)).copy()
    result = reference_pid(e, gains)
    np.testing.assert_allclose(result["I"], [0, 4, 8, 8, 8], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["u"], [0, 0.62, 0.44, -0.16, 0.04],
                               rtol=0, atol=1e-13)
    for sid, dat in sequences.items():
        if sid.startswith("sign_compensate"):
            if not np.isclose(dat["I"][-1], 0, rtol=0, atol=1e-9):
                raise AssertionError(f"Integral did not compensate in {sid}")
    hold = sequences["long_hold"]
    start = len(hold["e"])//8
    if not np.all(hold["e"][start:] == 0):
        raise AssertionError("long_hold needs an exact zero-input plateau")
    np.testing.assert_allclose(hold["I"][start:], hold["I"][start-1],
                               rtol=0, atol=1e-10)
    comp = sequences["long_z_comp"]
    n = len(comp["e"])
    if np.any(comp["e"][n//5:n//3] != 0):
        raise AssertionError("e must be zero when Ki changes")
    if np.ptp(comp["gains"][n//5:n//3, 1]) < 0.001:
        raise AssertionError("Ki must change noticeably during e=0")
    if abs(comp["I"][-1]) > 1e-9:
        raise AssertionError("long_z_comp must end with zero net integral")
    return {"status": "PASS", "tests": [
        "nominal coefficients at z=0", "numeric five-step PID reference",
        "q=4 coefficient example and all 16 corners (when applicable)",
        "sign-compensation integral", "long_hold state retention",
        "Ki changes while error is zero", "long_z_comp final compensation"]}


def stats_for(sequences: dict, specs: list[dict]) -> dict:
    merged = {}
    for name in ("e", "Kp", "Ki", "Kd", "P", "I", "D", "u_P", "u_I", "u_D", "u"):
        if name in ("Kp", "Ki", "Kd"):
            ix = ("Kp", "Ki", "Kd").index(name)
            arr = np.concatenate([sequences[s["sequence_id"]]["gains"][:, ix] for s in specs])
        else:
            arr = np.concatenate([sequences[s["sequence_id"]][name] for s in specs])
        merged[name] = {"min": float(np.min(arr)), "max": float(np.max(arr)),
                        "mean": float(np.mean(arr)), "std": float(np.std(arr))}
    return merged


def render_plot(path: Path, t: np.ndarray, items: list[tuple[str, np.ndarray]],
                title: str, ylabel: str) -> None:
    fig = plt.figure(figsize=(11, 4.2))
    ax = fig.add_subplot(111)
    for label, values in items:
        ax.plot(t, values, label=label, linewidth=1.35)
    ax.set(xlabel="Час, хв", ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.25)
    if len(items) > 1:
        ax.legend(loc="best", fontsize=8, ncol=min(4, len(items)))
    fig.tight_layout()
    fig.savefig(path, dpi=145)
    plt.close(fig)


def make_plots(folder: Path, specs: list[dict], sequences: dict, q: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    # Inspect every agreed error-signal family (one short example each).
    for kind in SIGNALS:
        sid = f"{kind}_A"
        dat = sequences[sid]
        time = np.arange(len(dat["e"])) * TS / 60.0
        render_plot(folder / f"signal_{kind}.png", time,
                    [("e", dat["e"])], f"Тип похибки: {kind}", "e")
    # Visual evidence that every Pilot "step" z-mode contains a real transition.
    for spec in specs:
        if spec["z_mode"] != "step":
            continue
        sid = spec["sequence_id"]
        dat = sequences[sid]
        t = np.arange(len(dat["e"])) * TS / 60.0
        render_plot(folder / f"step_z_{sid}.png", t,
                    [(f"z_{j+1}", dat["z"][:, j]) for j in range(q)],
                    sid + " — перевірка переходів z", "z")
    for sid in ("combined_B", "long_z_comp"):
        dat = sequences[sid]
        t = np.arange(len(dat["e"])) * TS / 60.0
        prefix = folder / sid
        render_plot(Path(f"{prefix}_01_error.png"), t, [("e", dat["e"])], sid+" — похибка", "e")
        render_plot(Path(f"{prefix}_02_z.png"), t,
                    [(f"z_{j+1}", dat["z"][:, j]) for j in range(q)],
                    sid+" — додаткові параметри", "z")
        for col, gain in enumerate(("Kp", "Ki", "Kd"), start=3):
            render_plot(Path(f"{prefix}_{col:02d}_{gain}.png"), t,
                        [(gain, dat["gains"][:, col-3])], sid+f" — {gain}", gain)
        render_plot(Path(f"{prefix}_06_pid_parts.png"), t,
                    [(name, dat[name]) for name in ("u_P", "u_I", "u_D")],
                    sid+" — складові ПІД", "Внески до u")
        render_plot(Path(f"{prefix}_07_u.png"), t, [("u", dat["u"])],
                    sid+" — керувальний сигнал", "u (без обмеження)")


def run(q: int, seed: int, out: Path) -> None:
    if q < 1:
        raise ValueError("q must be at least 1")
    out.mkdir(parents=True, exist_ok=True)
    weights = coefficient_weights(q, seed + 111)
    specs = build_specs(seed)
    sequences = {s["sequence_id"]: make_sequence(s, q, weights) for s in specs}

    write_data(out / "pilot.csv", specs, sequences, q)
    write_sequence_index(out / "sequences.csv", specs)
    config = {
        "dataset_version": "PID Dataset v1.1 — Pilot (coverage corrections)",
        "Ts_s": TS, "q": q, "master_seed": seed,
        "coefficient_weights_seed": seed + 111,
        "coefficient_weights_rows_Kp_Ki_Kd": weights.tolist(),
        "nominal_Kp_Ki_Kd": NOMINAL.tolist(), "relative_spans": REL_SPAN.tolist(),
        "coefficient_limits_Kp_Ki_Kd": LIMITS.tolist(),
        "equations": {"P": "e[k]", "I": "I[k-1]+Ts*e[k]",
                      "D": "(e[k]-e[k-1])/Ts",
                      "u": "Kp*P+Ki*I+Kd*D"},
        "initial_state": {"I_minus_1": 0.0, "e_minus_1": 0.0},
        "floating_point": "float64", "output_saturation": None,
        "derivative_filter": None, "anti_windup": False,
        "signal_types_short": list(SIGNALS), "sequences": specs,
        "coverage_policy": {"min_step_transitions_per_step_sequence": 1,
                            "first_z_step_window_fraction": [0.20, 0.60],
                            "later_z_step_dwells_samples": [180, 1080],
                            "min_observed_gain_range_fraction": 0.20,
                            "min_short_dominance_percent_each_PID": 5.0,
                            "all_and_long_dominance_are_diagnostic_only": True,
                            "q_1_weight_nonproportionality": "not applicable"},
        "pilot_simplifications": [
            "Normalized e and z in [-1,1], with sampled values not forced to the endpoints.",
            "Only linear, time-invariant K(z) mapping; no actual plant or closed-loop dynamics.",
            "16 short cases (2 per signal type), 2 long memory cases; not an exhaustive input coverage.",
            "Step z sequences must switch once early; later intervals remain slow.",
            "No measurement quantization, realistic sensor noise model, actuator model or hardware limits.",
            "Integrator has no anti-windup or reset inside an experiment. No output clipping."
        ],
        "planned_full_expansion_not_generated": [
            "More independent time series and stronger coverage of z corners, not just sampled interior.",
            "Cover rare long-duration bias/zero-error holds and delayed reversals; test over 24h.",
            "Expand amplitude/frequency/slope combinations and parameter correlations.",
            "Keep train/validation/test split at whole-sequence level; do not reuse seeds."
        ]
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    def safely(fn):
        try:
            return fn()
        except Exception as exc:
            return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}

    mathematical = safely(lambda: check_data(out / "pilot.csv", specs, weights, q))
    coverage = safely(lambda: check_coverage(out / "pilot.csv", specs, weights, q))
    tests = safely(lambda: unit_tests(weights, q, sequences))
    overall = "PASS" if all(x.get("status") == "PASS" for x in (mathematical, coverage, tests)) else "FAIL"
    report = {"overall_status": overall,
              "mathematical_validation": mathematical,
              "coverage_validation": coverage,
              "unit_tests": tests,
              "summary": stats_for(sequences, specs)}
    (out / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    make_plots(out / "plots", specs, sequences, q)

    print(f"Pilot v1.1: 18 independent sequences, 8192 rows; q={q}, Ts={TS:g}s")
    print(f"MATHEMATICS: {mathematical['status']}; COVERAGE: {coverage['status']}; UNIT TESTS: {tests['status']}")
    print(f"OVERALL: {overall}")
    if mathematical["status"] == "PASS":
        print(f"  max residual={mathematical['max_absolute_equation_residual']:.3g}; "
              f"|u| max={mathematical['max_absolute_u']:.6g}")
    if coverage["status"] == "PASS":
        print("  step z transition counts:", coverage["step_transition_counts"])
        for name, result in coverage["dominance_by_duration"].items():
            print(f"  {name:>5} P/I/D dominance (%):", {
                part: round(value, 2) for part, value in result["dominating_percent"].items()})
    else:
        print("  coverage failures:", [name for name, result in
              coverage.get("checks", {}).items() if result["status"] != "PASS"])
    print("  output:", out)
    if overall != "PASS":
        raise RuntimeError(f"Pilot validation failed; see {out/'validation_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and verify PID Dataset v1.0 Pilot only")
    parser.add_argument("--q", type=int, default=4, help="number of additional z inputs")
    parser.add_argument("--seed", type=int, default=20261102, help="master seed")
    parser.add_argument("--out", type=Path, default=Path("pid_pilot"), help="output folder")
    args = parser.parse_args()
    run(args.q, args.seed, args.out)
