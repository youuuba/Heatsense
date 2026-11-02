"""Command line.

    heatsense calibrate runs/*.fit --out profile.json   # learn your profile from past runs
    heatsense analyze run.fit --profile profile.json     # HeatSense vs the reference baseline
    heatsense demo                                        # simulated runner, known truth
    heatsense evaluate                                    # head-to-head on simulated athletes
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from .baseline import run_baseline
from .effort import minute_effort
from .io_fit import read_fit, to_minutes
from .metrics import compare
from .model import PhysicsParams, Profile, build_features, calibrate, run_heatsense, strain_zones
from .plot import plot_run
from .quality import check_hr, summarize
from .weather import run_weather


def expand_paths(paths):
    """Expand wildcards ourselves: PowerShell and cmd pass `runs/*.fit` through literally."""
    out = []
    for p in paths:
        matches = sorted(glob.glob(str(p))) if any(c in str(p) for c in "*?[") else [str(p)]
        if not matches:
            raise SystemExit(f"No files match {p}")
        out.extend(Path(m) for m in matches)
    # Windows globs ignore case, so `*.fit *.FIT` can list the same run twice
    unique = {}
    for p in out:
        unique.setdefault(p.resolve(), p)
    return list(unique.values())


def run_location(seconds):
    """Median GPS position rounded to ~1 km, or None (treadmill / no GPS)."""
    if "lat" not in seconds or seconds["lat"].isna().all():
        return None
    return round(float(seconds["lat"].median()), 2), round(float(seconds["lon"].median()), 2)


def prepare_run(seconds, weather: bool = True, fixed_wbgt: float | None = None,
                physics: PhysicsParams = PhysicsParams()):
    """Per-second data -> (checked seconds, minute table, model features, weather, note)."""
    location = run_location(seconds)
    seconds = seconds.drop(columns=["lat", "lon"], errors="ignore")  # never kept past here
    checked = check_hr(seconds)
    minutes = to_minutes(checked)
    wx = None
    if fixed_wbgt is not None:
        wbgt, note = fixed_wbgt, f"Heat stress set to WBGT {fixed_wbgt:.0f} °C."
    elif location is None:
        wbgt, note = physics.indoor_wbgt, f"Indoor / no GPS: assumed WBGT {physics.indoor_wbgt:.0f} °C."
    elif not weather:
        wbgt, note = physics.unknown_wbgt, f"Weather lookup skipped: assumed WBGT {physics.unknown_wbgt:.0f} °C."
    else:
        try:
            wx = run_weather(minutes, *location)
            wbgt = wx["wbgt_est"].to_numpy()
            moving = ~minutes["paused"]
            note = (f"Weather: {wx['air_temp'][moving].mean():.0f} °C, {wx['rh'][moving].mean():.0f}% RH, "
                    f"WBGT ≈ {wx['wbgt_est'][moving].mean():.0f} °C (Open-Meteo, location rounded to ~1 km).")
        except Exception as exc:  # network down, API change: analysis still runs
            wbgt = physics.unknown_wbgt
            note = f"Weather unavailable ({type(exc).__name__}): assumed WBGT {wbgt:.0f} °C."
    feat = build_features(minutes, minute_effort(checked, len(minutes)), wbgt, physics)
    return checked, minutes, feat, wx, note


def _analyze(seconds, title, out_png, out_csv, profile: Profile, ref_col=None,
             ref_label="Reference (core sensor)", weather=True, fixed_wbgt=None, note_override=None):
    checked, minutes, feat, wx, note = prepare_run(seconds, weather, fixed_wbgt)
    note = note_override or note
    hs = run_heatsense(feat, profile)
    zones = strain_zones(hs, feat)
    base = run_baseline(minutes["hr"])

    reference = None
    if ref_col and ref_col in checked and checked[ref_col].notna().any():
        elapsed = ((checked.index - checked.index[0]).total_seconds() // 60).astype(int).to_numpy()
        reference = checked[ref_col].groupby(elapsed).mean().reindex(minutes.index)

    plot_run(checked, minutes, hs, reference, ref_label, title, note=note, path=out_png,
             baseline=base, zones=zones, profile_label=profile.source)

    table = minutes.join(feat[["met_power", "wbgt"]]).join(hs).join(zones)
    table = table.join(base.rename(columns={"ct": "baseline_ct", "ct_sd": "baseline_sd", "gain": "baseline_gain"}))
    if wx is not None:
        table = table.join(wx.drop(columns=["wbgt_est"]))
    if reference is not None:
        table["reference_ct"] = reference
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        table.drop(columns="time").to_csv(out_csv, float_format="%.3f")

    q = summarize(checked)
    print(f"{title}: {len(minutes)} min ({int(minutes['paused'].sum())} paused), usable heart rate {q['usable']:.1%}")
    print(f"  {note}")
    for name, share in q.items():
        if name != "usable" and share > 0:
            print(f"  flagged {name}: {share:.1%}")
    print(f"  HeatSense peak {hs['ct'].max():.2f} °C, zones {zones.value_counts().to_dict()} "
          f"| reference baseline peak {base['ct'].max():.2f} °C")
    if reference is not None:
        for name, est in (("HeatSense", hs["ct"]), ("baseline", base["ct"])):
            m = compare(est, reference, skip_minutes=10)
            print(f"  {name} vs reference (after 10 min): n={m['n']}, bias {m['bias']:+.2f} °C, "
                  f"RMSE {m['rmse']:.2f} °C, 95% LoA [{m['loa_low']:+.2f}, {m['loa_high']:+.2f}]")
    print(f"  plot -> {out_png}" + (f", table -> {out_csv}" if out_csv else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="heatsense", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="Learn your personal profile from past runs")
    c.add_argument("fits", type=Path, nargs="+")
    c.add_argument("--out", type=Path, default=Path("profile.json"))
    c.add_argument("--no-weather", action="store_true")

    a = sub.add_parser("analyze", help="Analyze a Garmin .FIT run")
    a.add_argument("fit", type=Path)
    a.add_argument("--profile", type=Path, help="Profile from `heatsense calibrate` (default: population)")
    a.add_argument("--out", type=Path, help="PNG path (default: next to the .fit)")
    a.add_argument("--csv", type=Path, help="Per-minute table CSV path")
    a.add_argument("--no-weather", action="store_true", help="Skip the weather lookup")
    a.add_argument("--wbgt", type=float, help="Set heat stress (WBGT, °C) instead of looking it up")

    d = sub.add_parser("demo", help="Simulated runner with known core temperature")
    d.add_argument("--out", type=Path, default=Path("demo.png"))
    d.add_argument("--csv", type=Path)

    sub.add_parser("evaluate", help="HeatSense vs reference baseline on simulated athletes")

    args = parser.parse_args(argv)
    if args.cmd == "calibrate":
        feats = []
        for f in expand_paths(args.fits):
            _, _, feat, _, note = prepare_run(read_fit(f, keep_gps=True), weather=not args.no_weather)
            print(f"  {f.name}: {len(feat)} min. {note}")
            feats.append(feat)
        prof = calibrate(feats)
        prof.save(args.out)
        print(f"Profile ({prof.n_runs} runs, {prof.n_minutes} minutes) -> {args.out}")
        print(f"  heart rate ≈ {prof.hr_base:.0f} + {prof.hr_per_effort:.1f} × effort (W/kg); "
              f"max HR {prof.hr_max:.0f}; noise {prof.hr_noise_sd:.1f} bpm")
        for n in prof.notes:
            print(f"  note: {n}")
    elif args.cmd == "analyze":
        prof = Profile.load(args.profile) if args.profile else Profile()
        seconds = read_fit(args.fit, keep_gps=True)
        out = args.out or args.fit.with_suffix(".png")
        _analyze(seconds, f"Run on {seconds.index[0]:%Y-%m-%d}", out, args.csv, prof, "core_temp",
                 weather=not args.no_weather, fixed_wbgt=args.wbgt)
    elif args.cmd == "demo":
        from .evaluate import prepare
        from .synthetic import make_athlete_run
        # A runner who heats up 20% faster than typical, on a hot day: a harder case than
        # average. `heatsense evaluate` gives the averages across simulated athletes.
        gain, offset = 1.2, 8.0
        cal = [prepare(9000 + i, gain, offset)["feat"] for i in range(6)]
        prof = calibrate(cal)
        seed = next(s for s in range(9100, 9400)
                    if make_athlete_run(s, minutes=80, heat_gain=gain, hr_offset=offset)[1]["wbgt"] >= 27
                    and not make_athlete_run(s, minutes=80, heat_gain=gain, hr_offset=offset)[1]["indoor"])
        seconds, info = make_athlete_run(seed, minutes=80, heat_gain=gain, hr_offset=offset)
        _analyze(seconds, "Simulated hot run, known core temperature", args.out, args.csv, prof,
                 "true_ct", "Simulated true core temperature", fixed_wbgt=info["wbgt"],
                 note_override=f"Simulated runner who heats up 20% faster than typical; {info['air']:.0f} °C air, "
                               f"WBGT {info['wbgt']:.0f} °C. Profile calibrated on 6 other simulated runs.")
    else:
        from .evaluate import evaluate_athletes
        r = evaluate_athletes()
        cols = ["bias_C", "rmse_C", "hot_minutes_caught_(>=38.8)", "false_hot_alarms"]
        print(r.groupby("method")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
