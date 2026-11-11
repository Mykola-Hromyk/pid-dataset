"""PID Dataset v1.1 Full — generate, audit and plot the agreed 388 sequences.

Builds directly on the verified PID Dataset v1.1 Pilot functions; keep
pid_dataset_pilot_v1_1.py beside this file.

Usage:
    python pid_dataset_full_v1_1.py --q 4 --seed 20261102 --out pid_full_v1_1

Requires Python >=3.10, numpy, matplotlib. Does not train neural networks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pid_dataset_pilot_v1_1 as pilot

TS = pilot.TS
Q4_WEIGHTS = pilot.WEIGHTS_Q4
SIGNALS = pilot.SIGNALS
SPLITS = {
    "train": {"short": 240, "short_len": 1024, "long": 12, "long_len": 4096},
    "validation": {"short": 60, "short_len": 1024, "long": 4, "long_len": 4096},
    "test": {"short": 60, "short_len": 1024, "long": 12, "long_len": 8640},
}
EXPECTED_COUNTS = {"train": 294912, "validation": 77824, "test": 165120}
FLOAT_ATOL = pilot.CHECK_ATOL
FLOAT_RTOL = pilot.CHECK_RTOL


def corners_for_weights(q: int, weights: np.ndarray, seed: int, budget: int = 16) -> list[list[float]]:
    """All corners for q<=4; for q>=5, selected corners incl. each gain extremum."""
    if q <= 4:
        return [list(map(float, corner)) for corner in itertools.product((-1.0, 1.0), repeat=q)]
    chosen = set()
    for row in weights:
        sign = tuple(float(x) for x in np.where(row >= 0, 1.0, -1.0))
        chosen.add(sign)
        chosen.add(tuple(-x for x in sign))
    rng = np.random.default_rng(seed)
    while len(chosen) < budget:
        chosen.add(tuple(map(float, rng.choice(np.array([-1., 1.]), size=q))))
    return [list(v) for v in sorted(chosen)[:budget]]


def make_full_specs(seed: int, weights: np.ndarray, q: int) -> dict[str, list[dict]]:
    """Deterministic, disjoint splits with all eight e families and three z modes."""
    master = np.random.default_rng(seed)
    full = {}
    for split, sizes in SPLITS.items():
        ns, nl = sizes["short"], sizes["long"]
        offset = 4 if split == "test" else 0  # complementary 7/8 counts for val/test
        specs = [
            {"sequence_id": f"{split}_short_{idx:03d}_{SIGNALS[(idx+offset)%8]}",
             "signal_type": SIGNALS[(idx+offset)%8], "length": sizes["short_len"],
             "split": split, "kind": "short"}
            for idx in range(ns)
        ]
        master.shuffle(specs)
        for idx in range(nl):
            kind = "long_hold" if idx % 2 == 0 else "long_z_comp"
            specs.append({"sequence_id": f"{split}_long_{idx:03d}_{kind}",
                          "signal_type": kind, "length": sizes["long_len"],
                          "split": split, "kind": "long",
                          "z_mode": "constant" if kind == "long_hold" else "smooth"})

        totals = {
            "constant": int(round((ns+nl) * 0.25)),
            "step": int(round((ns+nl) * 0.15)),
        }
        totals["smooth"] = ns + nl - totals["constant"] - totals["step"]
        modes = (["constant"] * (totals["constant"] - nl//2)
                 + ["smooth"] * (totals["smooth"] - nl//2)
                 + ["step"] * totals["step"])
        assert len(modes) == ns and min(totals.values()) > 0
        master.shuffle(modes)
        for spec, mode in zip(specs[:ns], modes):
            spec["z_mode"] = mode

        # Reserve actual corners in EVERY split, rather than hoping a random
        # trajectory eventually reaches them. This does not clip z, K or u.
        corner_targets = corners_for_weights(q, weights, seed + 1503)
        constant_specs = [s for s in specs if s["z_mode"] == "constant"]
        assert len(constant_specs) >= len(corner_targets)
        master.shuffle(constant_specs)
        for corner_id, (spec, corner) in enumerate(zip(constant_specs, corner_targets)):
            spec["corner_z"] = corner
            spec["corner_id"] = corner_id

        # Guarantee both exact normalized extrema of e in every independent
        # split. This is a carefully designed INPUT waveform, not PID clipping.
        selected = next(s for s in specs if s["signal_type"] == "prbs_pulses")
        selected["force_e_extremes"] = True

        # Every error and z sequence has its own reproducible random source.
        for spec in specs:
            spec["error_seed"] = int(master.integers(0, 2**32, dtype=np.uint32))
            spec["z_seed"] = int(master.integers(0, 2**32, dtype=np.uint32))
            spec["amplitude_seed"] = int(master.integers(0, 2**32, dtype=np.uint32))
        full[split] = specs
    return full


def full_sequence(spec: dict, q: int, weights: np.ndarray) -> dict:
    """Reuse Pilot's approved e, z, gain and PID computations unchanged."""
    data = pilot.make_sequence(spec, q, weights)
    e, z = data["e"], data["z"]
    updated = False
    if spec.get("corner_z") is not None:
        z = np.broadcast_to(np.asarray(spec["corner_z"], dtype=np.float64),
                            (spec["length"], q)).copy()
        updated = True
    if spec.get("force_e_extremes"):
        e = e.copy()
        e[spec["length"]//4] = np.float64(1.0)
        e[3*spec["length"]//4] = np.float64(-1.0)
        updated = True
    if spec["kind"] == "long":
        # Various magnitudes/signs keep the long-run tests independent, while
        # preserving their exact zero-error holds and sign compensation.
        rng = np.random.default_rng(spec["amplitude_seed"])
        scale = float(rng.uniform(0.60, 1.25)) * float(rng.choice([-1.0, 1.0]))
        e = np.asarray(e * scale, dtype=np.float64)
        updated = True
    if updated:
        gains = pilot.coefficients(z, weights)
        data = {"e": e, "z": z, "gains": gains, **pilot.reference_pid(e, gains)}
    return data


def write_index(path: Path, specs: dict[str, list[dict]]) -> None:
    headers = ("split", "sequence_id", "kind", "signal_type", "z_mode", "length",
               "error_seed", "z_seed", "amplitude_seed", "corner_id", "corner_z",
               "force_e_extremes")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for split in SPLITS:
            for s in specs[split]:
                writer.writerow([json.dumps(s.get(k), ensure_ascii=False, separators=(',', ':'))
                                 if k == "corner_z" and k in s else s.get(k, "") for k in headers])


def diagnose_and_validate(path: Path, specs: list[dict], q: int,
                          weights: np.ndarray, split: str) -> dict:
    """Independent STREAMING re-read from CSV, vectorized identities per sequence.

    This rechecks written data, not just in-memory generator arrays. At most
    one sequence (8640 rows) is held in memory during validation.
    """
    fields = pilot.columns(q)
    numeric_names = fields[5:]
    numeric_offsets = {name: idx for idx, name in enumerate(numeric_names)}
    rows_total = 0
    max_residual = 0.0
    zero_hold_count = 0
    minmax = {name: [float('inf'), float('-inf')] for name in
              ("e", "I", "D", "Kp", "Ki", "Kd", "u")}
    z_min = np.full(q, np.inf)
    z_max = np.full(q, -np.inf)
    z_mode_counts = Counter()
    signal_counts = Counter()
    step_switches = {}
    smooth_changes = {}
    constant_max_diffs = {}
    corners_actual = set()
    required_corners = set()
    group_stats = {name: {"informative": 0, "dominant": np.zeros(3, dtype=np.int64),
                          "active": np.zeros(3, dtype=np.int64), "rows": 0}
                   for name in ("short", "long", "all")}

    def check_eq(actual: np.ndarray, expected: np.ndarray, name: str, sid: str) -> None:
        nonlocal max_residual
        residual = np.abs(actual - expected)
        local_max = float(np.max(residual)) if residual.size else 0.0
        max_residual = max(max_residual, local_max)
        if not np.allclose(actual, expected, atol=FLOAT_ATOL, rtol=FLOAT_RTOL):
            index = int(np.argmax(residual))
            raise AssertionError(f"{split}/{sid}: {name} at k={index} actual={actual[index]} "
                                 f"expected={expected[index]} err={residual[index]}")

    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if next(reader) != fields:
            raise AssertionError(f"{split}: CSV schema changed")
        for s in specs:
            sid, n = s["sequence_id"], s["length"]
            metadata = []
            values = []
            for k in range(n):
                row = next(reader, None)
                if row is None or len(row) != len(fields):
                    raise AssertionError(f"{split}/{sid}: incomplete CSV at k={k}")
                if row[0] != sid or int(row[1]) != k or int(row[2]) != int(k*TS):
                    raise AssertionError(f"{split}/{sid}: sequence/time mismatch at k={k}")
                if row[3] != s["signal_type"] or row[4] != s["z_mode"]:
                    raise AssertionError(f"{split}/{sid}: signal/z mode mismatch")
                values.append([float(v) for v in row[5:]])
            data = np.asarray(values, dtype=np.float64)
            if not np.isfinite(data).all():
                raise AssertionError(f"{split}/{sid}: nonfinite numeric value")
            num = lambda name: data[:, numeric_offsets[name]]
            e = num("e")
            z = data[:, 1:1+q]
            gains = data[:, 1+q:4+q]
            p, integ, d = (num(name) for name in ("P", "I", "D"))
            up, ui, ud, u = (num(name) for name in ("u_P", "u_I", "u_D", "u"))
            if np.max(np.abs(e)) > 1+1e-12 or np.max(np.abs(z)) > 1+1e-12:
                raise AssertionError(f"{split}/{sid}: inputs outside [-1,1]")
            expected_gains = pilot.coefficients(z, weights)
            check_eq(gains, expected_gains, "K(z)", sid)
            if not ((gains >= pilot.LIMITS[:, 0]-1e-12) &
                    (gains <= pilot.LIMITS[:, 1]+1e-12)).all():
                raise AssertionError(f"{split}/{sid}: coefficient bounds exceeded")
            prev_e = np.r_[0.0, e[:-1]]
            prev_i = np.r_[0.0, integ[:-1]]
            check_eq(p, e, "P", sid)
            check_eq(integ, prev_i + TS*e, "stateful I", sid)
            check_eq(d, (e-prev_e)/TS, "D", sid)
            for name, actual, expected in (("u_P", up, gains[:, 0]*p),
                                            ("u_I", ui, gains[:, 1]*integ),
                                            ("u_D", ud, gains[:, 2]*d),
                                            ("u", u, up+ui+ud)):
                check_eq(actual, expected, name, sid)
            zero_hold = (e[:-1] == 0) & (e[1:] == 0)
            if np.any(zero_hold):
                check_eq(integ[1:][zero_hold], integ[:-1][zero_hold], "zero-error hold", sid)
                zero_hold_count += int(np.count_nonzero(zero_hold))
            rows_total += n
            z_mode_counts[s["z_mode"]] += 1
            if s["kind"] == "short":
                signal_counts[s["signal_type"]] += 1
            for name in minmax:
                x = num(name)
                minmax[name][0] = min(minmax[name][0], float(x.min()))
                minmax[name][1] = max(minmax[name][1], float(x.max()))
            z_min = np.minimum(z_min, z.min(axis=0))
            z_max = np.maximum(z_max, z.max(axis=0))
            diffs = np.max(np.abs(np.diff(z, axis=0)), axis=1)
            switches = int(np.count_nonzero(diffs > 1e-10))
            if s["z_mode"] == "step":
                step_switches[sid] = switches
            elif s["z_mode"] == "smooth":
                smooth_changes[sid] = switches
            else:
                constant_max_diffs[sid] = float(np.max(diffs))
                if np.all(np.isin(z[0], (-1.0, 1.0))):
                    corners_actual.add(tuple(z[0].tolist()))
            if "corner_z" in s:
                corner = np.asarray(s["corner_z"], dtype=np.float64)
                required_corners.add(tuple(corner.tolist()))
                if not np.array_equal(z, np.broadcast_to(corner, z.shape)):
                    raise AssertionError(f"{split}/{sid}: designated corner z not constant")
            magnitudes = np.abs(np.stack((up, ui, ud), axis=1))
            total = magnitudes.sum(axis=1)
            mask = total > 1e-8
            ratios = np.zeros_like(magnitudes)
            ratios[mask] = magnitudes[mask] / total[mask, None]
            dominant = (np.argmax(magnitudes, axis=1)[:, None] ==
                        np.arange(3)[None, :]) & (ratios > 0.5) & mask[:, None]
            for group in (s["kind"], "all"):
                group_stats[group]["rows"] += n
                group_stats[group]["informative"] += int(np.count_nonzero(mask))
                group_stats[group]["dominant"] += np.sum(dominant, axis=0)
                group_stats[group]["active"] += np.sum(magnitudes > 1e-8, axis=0)
        extra = next(reader, None)
        if extra is not None:
            raise AssertionError(f"{split}: extra CSV records after final sequence")

    total_expected = EXPECTED_COUNTS[split]
    math_checks = {
        "csv_rows_exact": rows_total == total_expected,
        "sequence_count_exact": len(specs) == (SPLITS[split]["short"]+SPLITS[split]["long"]),
        "initial_states_and_recurrence": True,  # every sequence checked from zero independently
        "coefficients_and_pid_algebra": True,
        "float64_finite_and_input_bounds": True,
    }
    dominance = {}
    for name, values in group_stats.items():
        ninf = values["informative"]
        dominance[name] = {
            "rows": values["rows"], "informative_rows": ninf,
            "dominating_percent": {key: round(100*int(values["dominant"][idx])/ninf, 4)
                                   if ninf else 0.0 for idx, key in enumerate(("P", "I", "D"))},
            "active_nonzero_rows": {key: int(values["active"][idx])
                                    for idx, key in enumerate(("P", "I", "D"))},
        }

    corner_target = min(2**q, 16) if q <= 4 else 16
    coefficient_extrema = {}
    for gain in ("Kp", "Ki", "Kd"):
        actual = minmax[gain]
        j = ("Kp", "Ki", "Kd").index(gain)
        lo, hi = pilot.LIMITS[j]
        coefficient_extrema[gain] = {"min": actual[0], "max": actual[1],
                                      "theoretical_min": float(lo), "theoretical_max": float(hi),
                                      "both_endpoints_hit": bool(abs(actual[0]-lo)<1e-11 and abs(actual[1]-hi)<1e-11)}
    coverage_checks = {
        "all_e_families_each_split": all(signal_counts.get(s,0) >= 2 for s in SIGNALS),
        "all_z_modes_each_split": all(z_mode_counts.get(z,0) > 0 for z in ("constant", "smooth", "step")),
        "every_step_z_has_transition": bool(step_switches) and all(v>=1 for v in step_switches.values()),
        "constant_z_remains_constant": all(v<=1e-12 for v in constant_max_diffs.values()),
        "smooth_z_changes": bool(smooth_changes) and all(v>0 for v in smooth_changes.values()),
        "both_e_endpoints": abs(minmax["e"][0]+1)<1e-12 and abs(minmax["e"][1]-1)<1e-12,
        "all_z_coordinate_endpoints": bool(np.all(np.abs(z_min+1)<1e-12) and np.all(np.abs(z_max-1)<1e-12)),
        "all_gain_endpoints": all(g["both_endpoints_hit"] for g in coefficient_extrema.values()),
        "designated_z_corners_present": (len(required_corners)==corner_target and
                                         required_corners.issubset(corners_actual)),
        "short_PID_dominance_ge_5pct": all(v>=5.0 for v in dominance["short"]["dominating_percent"].values()),
        "long_PID_nonzero_activity": all(v>0 for v in dominance["long"]["active_nonzero_rows"].values()),
        "zero_error_hold_in_split": zero_hold_count>0,
        "unclipped_u_evidence": max(abs(x) for x in minmax["u"])>1.0,
    }
    # numpy comparisons can yield np.bool_ (not supported by the standard JSON encoder).
    coverage_checks = {key: bool(value) for key, value in coverage_checks.items()}
    mat_status = "PASS" if all(math_checks.values()) else "FAIL"
    coverage_status = "PASS" if all(coverage_checks.values()) else "FAIL"
    return {
        "mathematical_validation": {"status": mat_status, "checks": math_checks,
                                    "rows": rows_total, "sequences": len(specs),
                                    "max_equation_absolute_residual": max_residual,
                                    "zero_error_hold_checks": zero_hold_count},
        "coverage_validation": {"status": coverage_status, "checks": coverage_checks,
                                 "short_signal_families": dict(signal_counts),
                                 "z_modes": dict(z_mode_counts),
                                 "z_step_transitions": step_switches,
                                 "z_step_count_min": min(step_switches.values()),
                                 "corner_target_count": corner_target,
                                 "corners_observed": len(corners_actual),
                                 "designated_corners_observed": len(required_corners),
                                 "coefficient_endpoints": coefficient_extrema,
                                 "observed_ranges": {key: {"min": v[0], "max": v[1]}
                                                     for key, v in minmax.items()},
                                 "z_by_coordinate": [{"min": float(z_min[j]), "max": float(z_max[j])}
                                                      for j in range(q)],
                                 "dominance_by_duration": dominance,
                                 "policy": {"short_dominance_threshold_pct": 5.0,
                                            "long_PID_dominance_is_diagnostic_only": True,
                                            "gain_endpoints_hit_by_designed_z_corners": True}}
    }


def json_numpy_default(value):
    """Convert only NumPy scalar/array report types to ordinary JSON types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1<<20), b""):
            h.update(block)
    return h.hexdigest()


def summary_plots(out: Path, specs_by_split: dict, representative: dict, q: int) -> None:
    folder = out / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    # One representative plot for each of the eight agreed error families.
    for signal in SIGNALS:
        spec = next(s for s in specs_by_split["train"] if s["signal_type"] == signal)
        dat = representative[spec["sequence_id"]]
        t = np.arange(len(dat["e"])) * TS / 60.0
        pilot.render_plot(folder / f"error_{signal}.png", t, [("e", dat["e"])],
                          "Full — " + signal, "e")
    # Full diagnostics in each split, including the long memory case and a z step.
    for split, specs in specs_by_split.items():
        chosen = [next(s for s in specs if s["signal_type"] == "long_z_comp"),
                  next(s for s in specs if s["z_mode"] == "step")]
        for spec in chosen:
            dat = representative[spec["sequence_id"]]
            sid = spec["sequence_id"]
            t = np.arange(len(dat["e"])) * TS / 60.0
            pilot.render_plot(folder / f"{sid}_e.png", t, [("e", dat["e"])], sid+" — e", "e")
            pilot.render_plot(folder / f"{sid}_z.png", t,
                              [(f"z_{j+1}", dat["z"][:,j]) for j in range(q)], sid+" — z", "z")
            pilot.render_plot(folder / f"{sid}_gains.png", t,
                              [(name, dat["gains"][:,j]) for j,name in enumerate(("Kp", "Ki", "Kd"))],
                              sid+" — Kp, Ki, Kd", "K")
            pilot.render_plot(folder / f"{sid}_parts.png", t,
                              [(name,dat[name]) for name in ("u_P", "u_I", "u_D")],
                              sid+" — P/I/D внески", "u складові")
            pilot.render_plot(folder / f"{sid}_u.png", t, [("u", dat["u"])], sid+" — u", "u")


def run(q: int, seed: int, out: Path, plots: bool = True) -> dict:
    if q < 1:
        raise ValueError("q must be >=1")
    out.mkdir(parents=True, exist_ok=True)
    weights = pilot.coefficient_weights(q, seed+111)  # EXACT Pilot v1.1 mapping
    weight_report = pilot.check_weight_vectors(weights, q)
    all_specs = make_full_specs(seed, weights, q)
    all_ids = [s["sequence_id"] for part in all_specs.values() for s in part]
    all_seeds = [s[k] for part in all_specs.values() for s in part
                 for k in ("error_seed", "z_seed", "amplitude_seed")]
    if len(all_ids) != len(set(all_ids)) or len(all_seeds) != len(set(all_seeds)):
        raise AssertionError("Duplicate sequence ID or random seed across independent runs")
    if len(all_ids) != 388 or sum(s["length"] for part in all_specs.values() for s in part) != 537856:
        raise AssertionError("Full configuration does not equal 388/537856")
    write_index(out / "sequences.csv", all_specs)

    report = {"dataset_version": "PID Dataset v1.1 Full", "q": q, "Ts_s": TS,
              "master_seed": seed, "coefficient_weight_check": weight_report, "splits": {}}
    rep = {}
    for split in SPLITS:
        specs = all_specs[split]
        sequences = {s["sequence_id"]: full_sequence(s, q, weights) for s in specs}
        # Reuse the verified Pilot CSV schema and write function.
        path = out / f"{split}.csv"
        pilot.write_data(path, specs, sequences, q)
        if plots:
            for s in specs:
                if s["signal_type"] == "long_z_comp" and s["sequence_id"] not in rep:
                    rep[s["sequence_id"]] = sequences[s["sequence_id"]]
                    break
            for s in specs:
                if s["z_mode"] == "step":
                    rep[s["sequence_id"]] = sequences[s["sequence_id"]]
                    break
            if split == "train":
                for signal in SIGNALS:
                    s = next(s for s in specs if s["signal_type"] == signal)
                    rep[s["sequence_id"]] = sequences[s["sequence_id"]]
        del sequences
        status = diagnose_and_validate(path, specs, q, weights, split)
        report["splits"][split] = status
        print(f"{split:>10}: rows={EXPECTED_COUNTS[split]:,} / "
              f"math={status['mathematical_validation']['status']} / "
              f"coverage={status['coverage_validation']['status']}", flush=True)
        for kind, item in status["coverage_validation"]["dominance_by_duration"].items():
            print(f"  {kind:>5} P/I/D dominance (%): {item['dominating_percent']}", flush=True)

    report["cross_split_checks"] = {
        "status": "PASS",
        "sequence_ids_disjoint": len(set(all_ids))==388,
        "seeds_distinct": len(set(all_seeds))==len(all_seeds),
        "total_sequences": len(all_ids),
        "total_rows": sum(EXPECTED_COUNTS.values()),
        "weight_mapping_is_identical_across_splits": True,
    }
    for split in SPLITS:
        report["splits"][split]["sha256_csv"] = sha256(out / f"{split}.csv")
    report["overall_status"] = (
        "PASS" if all(v[x]["status"]=="PASS" for v in report["splits"].values()
                      for x in ("mathematical_validation", "coverage_validation")) else "FAIL")

    config = {
        "dataset_version": "PID Dataset v1.1 Full",
        "base_implementation": "pid_dataset_pilot_v1_1.py",
        "Ts_s": TS, "q": q, "master_seed": seed,
        "coefficient_weights_seed": seed+111,
        "coefficient_weights_rows_Kp_Ki_Kd": weights.tolist(),
        "nominal_Kp_Ki_Kd": pilot.NOMINAL.tolist(),
        "relative_spans": pilot.REL_SPAN.tolist(),
        "coefficient_limits_Kp_Ki_Kd": pilot.LIMITS.tolist(),
        "pid_law": {"P":"e[k]", "I":"I[k-1] + Ts*e[k]",
                    "D":"(e[k]-e[k-1])/Ts", "u":"Kp*P+Ki*I+Kd*D"},
        "initial_state_each_independent_sequence": {"I_minus_1":0.0, "e_minus_1":0.0},
        "floating_point_reference": "float64", "output_saturation": None,
        "derivative_filter": None, "anti_windup": False,
        "normalized_input_bounds": {"e":[-1.0, 1.0], "z_each":[-1.0,1.0]},
        "split_configuration": SPLITS,
        "expected_total": {"sequences":388,"samples":537856},
        "z_mode_targets": {"constant": "~25%", "smooth": "~60%", "step": "~15%"},
        "corner_strategy": "all 2^q corners if q<=4; otherwise 16 selected corners including the 6 gain extrema",
        "coverage_policy": {"all_splits_verify_e_z_and_gain_endpoints": True,
                             "short_dominance_min_each_pct":5.0,
                             "long_dominance_diagnostic_only":True,
                             "step_z_min_real_transitions":1},
        "pilot_functions_reused": ["coefficient_weights", "check_weight_vectors",
                                    "coefficients", "make_error", "make_z", "make_sequence",
                                    "reference_pid", "columns", "write_data", "render_plot"],
        "full_extensions": [
            "Sequence-level Train/Validation/Test separation; no shared sequence IDs or seeds.",
            "Intentional exact z-corner input trajectories in all three subsets; no gain clipping.",
            "A short PRBS waveform contains exact e=-1 and e=1 in each split.",
            "Every step-z sequence contains a real step, inherited from Pilot v1.1.",
            "Two types of long uninterrupted memory tests, with varied sign/amplitude.",
            "Mathematical correctness and coverage checked separately from the written CSV.",
        ],
        "remaining_limitations": [
            "Synthetic open-loop error histories do not prove closed-loop control stability.",
            "No actuator/sensor physics, delays, quantization, derivative filtering or anti-windup.",
            "q>4 does not exhaust all 2^q hypercube vertices; it covers gain extrema and selected corners.",
            "A 24-hour test cannot guarantee unlimited-duration integral fidelity.",
            "Long sequences emphasize integral; no 5% dominance requirement for long subset.",
            "Any normalization for neural training must be fit only to Train and must not reveal I/P/D reference inputs.",
        ],
        "sequences": all_specs,
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=json_numpy_default)+"\n", encoding="utf-8")
    (out / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_numpy_default)+"\n", encoding="utf-8")
    if plots:
        summary_plots(out, all_specs, rep, q)
    print("OVERALL:", report["overall_status"], flush=True)
    if report["overall_status"] != "PASS":
        failures = {split:[name for name,ok in result["coverage_validation"]["checks"].items() if not ok]
                    for split,result in report["splits"].items()}
        print("COVERAGE FAILURES:", failures, flush=True)
        raise RuntimeError(f"Full audit failed. See {out/'validation_report.json'}")
    return report


if __name__ == "__main__":
    argp = argparse.ArgumentParser(description="PID Dataset v1.1 FULL only (388 sequences)")
    argp.add_argument("--q", type=int, default=4)
    argp.add_argument("--seed", type=int, default=20261102)
    argp.add_argument("--out", type=Path, default=Path("pid_full_v1_1"))
    argp.add_argument("--no-plots", action="store_true", help="skip diagnostic PNG charts")
    args = argp.parse_args()
    run(args.q, args.seed, args.out, plots=not args.no_plots)
