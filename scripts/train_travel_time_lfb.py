#!/usr/bin/env python3
"""
Placement-ready London (LFB) drive-time trainer.

Data:   data/processed/lfb_travel_training.csv (scripts/build_lfb_travel_training.py):
        one row per engine trip from LFB mobilisation records.
Target: drive_s = recorded TravelTimeSeconds (leaving the station -> arriving),
        which LFB records separately from turnout; 30-1200 s.
Model:  gradient-boosted decision trees (LightGBM) predicting drive_s directly,
        averaged over 4 random seeds (42-45). No parametric prior: the Kolesar
        formula is refit on train only as a comparison baseline.

Features are placement-safe: road_km (shortest legal route, one-way streets
respected), crow_km, hour, dow, month, rush, night, busy_flag,
borough (label-encoded area context). Station ID / first_due_engine / station
name are deliberately NOT features, so the model can score hypothetical new
sites. The trip origin is the real building of the station the engine left
from (london_firehouses.csv, from OpenStreetMap), and feeds road_km and crow_km.

Chronological split: train 2021-2024, test 2025.

busy_flag (engine from outside the incident's station ground) is known
historically but must be assumed (e.g. expected busy rate) for a new site.

Run:  .venv/bin/python scripts/train_travel_time_lfb.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emvro.kolesar import fit_kolesar  # noqa: E402

DRIVE_CLIP_S = (30.0, 1200.0)
TRAIN_YEARS = (2021, 2022, 2023, 2024)

FEATURES = ["road_km", "crow_km", "hour", "dow", "month", "rush", "night", "busy_flag", "borough_code"]
SEEDS = (42, 43, 44, 45)


def add_features(df: pd.DataFrame, borough_cats: list[str]) -> pd.DataFrame:
    d = pd.to_datetime(df["date_of_call"], errors="coerce")
    out = pd.DataFrame(index=df.index)
    out["road_km"] = df["road_km"].to_numpy()
    out["crow_km"] = df["crow_km"].to_numpy()
    out["hour"] = df["hour_of_call"].astype(int).to_numpy()
    out["dow"] = d.dt.dayofweek.to_numpy()
    out["month"] = d.dt.month.to_numpy()
    h = out["hour"]
    out["rush"] = (h.between(7, 9) | h.between(16, 19)).astype(int)
    out["night"] = ((h >= 22) | (h <= 5)).astype(int)
    out["busy_flag"] = df["busy_flag"].astype(int).to_numpy()
    codes = {b: i for i, b in enumerate(borough_cats)}
    out["borough_code"] = df["borough"].astype(str).str.upper().map(codes).astype(float)  # unseen -> NaN
    return out[FEATURES]


def make_regressor(seed: int):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        objective="l1", random_state=seed, n_jobs=4, verbose=-1,
    )


def fit_bag(x: pd.DataFrame, y: np.ndarray):
    models = []
    for s in SEEDS:
        m = make_regressor(s)
        m.fit(x, y, categorical_feature=["borough_code"])
        models.append(m)
    return models


def predict_bag(models, x: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict(x) for m in models], axis=0)


def metrics(y, p):
    e = np.abs(y - p)
    return {"mae_s": float(e.mean()), "medae_s": float(np.median(e)), "bias_s": float(np.mean(p - y))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "lfb_travel_training.csv")
    ap.add_argument("--test-year", type=int, default=2025)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "models" / "travel_time_lfb.joblib")
    args = ap.parse_args()

    inc = pd.read_csv(args.data)
    train_years = [y for y in TRAIN_YEARS if (inc.cal_year == y).any()]
    tr = inc[inc.cal_year.isin(train_years)].reset_index(drop=True)
    te = inc[inc.cal_year == args.test_year].reset_index(drop=True)
    print(f"train years {train_years}: {len(tr):,} rows | test {args.test_year}: {len(te):,} rows")

    borough_cats = sorted(tr["borough"].astype(str).str.upper().unique())
    xtr, xte = add_features(tr, borough_cats), add_features(te, borough_cats)
    ytr, yte = tr["drive_s"].to_numpy(), te["drive_s"].to_numpy()

    # Kolesar is a comparison baseline only; the model below does not use it.
    kol = fit_kolesar(xtr["road_km"], ytr)
    kol_te = np.clip(kol.predict(xte["road_km"]), *DRIVE_CLIP_S)
    kol_crow = fit_kolesar(xtr["crow_km"], ytr)
    kol_crow_te = np.clip(kol_crow.predict(xte["crow_km"]), *DRIVE_CLIP_S)
    # Road distance at one constant speed, fit on train (total km / total seconds).
    road_kph = xtr["road_km"].sum() / ytr.sum() * 3600
    models = fit_bag(xtr, ytr)
    gbdt_te = np.clip(predict_bag(models, xte), *DRIVE_CLIP_S)

    results = {
        "median": metrics(yte, np.full_like(yte, np.median(ytr))),
        "crow@32kph": metrics(yte, np.clip(xte["crow_km"].to_numpy() / 32.0 * 3600, *DRIVE_CLIP_S)),
        f"road@{road_kph:.0f}kph": metrics(yte, np.clip(xte["road_km"].to_numpy() / road_kph * 3600, *DRIVE_CLIP_S)),
        "kolesar(crow_km)": metrics(yte, kol_crow_te),
        "kolesar(road_km)": metrics(yte, kol_te),
        f"lightgbm(x{len(SEEDS)} seeds)": metrics(yte, gbdt_te),
    }
    # Same models, non-busy subset (cleaner "own-ground" trips)
    nb = (xte["busy_flag"] == 0).to_numpy()
    results_nonbusy = {
        "kolesar": metrics(yte[nb], kol_te[nb]),
        "lightgbm": metrics(yte[nb], gbdt_te[nb]),
        "median": metrics(yte[nb], np.full(nb.sum(), np.median(ytr))),
    }

    print(f"\nHeld-out {args.test_year} driving-time error (recorded TravelTimeSeconds):")
    print(f"{'model':<28}{'MAE s':>9}{'MedAE s':>9}{'bias s':>9}")
    for k, v in results.items():
        print(f"{k:<28}{v['mae_s']:>9.1f}{v['medae_s']:>9.1f}{v['bias_s']:>9.1f}")
    print("non-busy subset only:", {k: round(v["mae_s"], 1) for k, v in results_nonbusy.items()})
    print("Kolesar on road_km (km,s):", {k: round(v, 3) for k, v in kol.to_dict().items()})

    payload = {
        "kind": "lightgbm",
        "models": models,
        "features": FEATURES,
        "categorical": ["borough_code"],
        "borough_categories": borough_cats,
        "kolesar_baseline_params": kol.to_dict(),
        "kolesar_baseline_distance": "road_km",
        "target": "drive_s = LFB TravelTimeSeconds (station departure -> arrival), 30-1200 s",
        "drive_clip_s": list(DRIVE_CLIP_S),
        "seeds": list(SEEDS),
        "train_years": train_years,
        "test_year": args.test_year,
        "metrics_test": results,
        "metrics_test_non_busy": results_nonbusy,
        "excluded_features": ["first_due_engine", "station name/ID (not placement-safe)"],
        "honesty_note": (
            "Origin = OSM location of the station each engine left from (LFB mobilisation records); "
            "engines already out on the road are excluded. No station ID feature so new sites can be "
            "scored; borough is only area context. busy_flag must be assumed for hypothetical sites. "
            "road_km = shortest-distance route on the OSM drive graph with one-way streets respected "
            "(not a learned route); no street characteristics yet."
        ),
    }
    import joblib

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.out)
    (args.out.with_suffix(".metrics.json")).write_text(
        json.dumps({k: v for k, v in payload.items() if k not in {"models"}}, indent=2, default=str)
    )
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
