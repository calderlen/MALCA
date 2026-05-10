#!/usr/bin/env python3
"""Test microlensing pipeline recovery rate against known ASAS-SN events.

Reads the catalog of known ASAS-SN microlensing events, fetches their
lightcurves, runs the fitting pipeline, and reports recovery statistics.

Usage:
    python scripts/test_microlensing_recovery.py [--max-events N] [--workers N] [--output DIR]

Output:
    - recovery_results.csv: Per-event results with fit parameters
    - recovery_summary.txt: Aggregate statistics
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from malca.fetch import cone_search, download_lightcurve_by_id
from malca.config import SKYPATROL_CACHE_DIR


def parse_asassn_microlens_csv(csv_path: Path) -> pd.DataFrame:
    """Parse the ASAS-SN microlensing events CSV.
    
    The CSV has an irregular format with quoted fields containing commas.
    We look for:
    - Event name patterns (ASASSN-*, AT*, Gaia*, OGLE-*)
    - Sexagesimal coordinates (HH:MM:SS, DD:MM:SS)
    - Decimal coordinates after the quoted description field
    """
    events = []
    
    with open(csv_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Skip header line
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        
        # Extract event name - look for known patterns
        name_match = re.search(r'(ASASSN-\d+\w*|AT\d{4}\w+|Gaia\d+\w+|OGLE-\d+-BLG-\d+)', line)
        if not name_match:
            continue
        name = name_match.group(1)
        
        # Strategy 1: Look for sexagesimal coordinates (HH:MM:SS.ss, -DD:MM:SS.ss)
        ra_str = None
        dec_str = None
        
        # RA is always positive, Dec can be negative
        sex_ra = re.findall(r'(\d{1,2}:\d{2}:\d{1,2}(?:\.\d+)?)', line)
        sex_dec = re.findall(r'(-?\d{1,2}:\d{2}:\d{1,2}(?:\.\d+)?)', line)
        
        if len(sex_ra) >= 1 and len(sex_dec) >= 2:
            ra_str = sex_ra[0]
            # Dec is the second match (first is RA without sign)
            dec_str = sex_dec[1]
        
        # Strategy 2: Look for decimal coordinates after quoted field
        # Pattern: "...",decimal_ra,decimal_dec
        if ra_str is None:
            dec_match = re.search(r'"[^"]*",\s*(\d+\.\d+),\s*(-?\d+\.\d+)', line)
            if dec_match:
                ra_str = dec_match.group(1)
                dec_str = dec_match.group(2)
        
        # Strategy 3: Look for any pair of decimals that could be RA/Dec
        if ra_str is None:
            # Split by comma, look for consecutive numeric values
            parts = line.split(',')
            for j in range(len(parts) - 1):
                try:
                    val1 = float(parts[j].strip())
                    val2 = float(parts[j + 1].strip())
                    # Check if they could be RA/Dec
                    if 0 < val1 < 360 and -90 < val2 < 90:
                        ra_str = str(val1)
                        dec_str = str(val2)
                        break
                except ValueError:
                    continue
        
        if ra_str is None or dec_str is None:
            continue
        
        # Parse coordinates
        try:
            if ':' in ra_str:
                # Handle negative declination in sexagesimal
                dec_sign = -1 if dec_str.startswith('-') else 1
                dec_str_clean = dec_str.lstrip('-')
                coord = SkyCoord(ra_str, dec_str_clean, unit=('hourangle', 'deg'))
                ra_deg = coord.ra.deg
                dec_deg = coord.dec.deg * dec_sign
            else:
                ra_deg = float(ra_str)
                dec_deg = float(dec_str)
        except Exception as e:
            continue
        
        # Validate coordinates
        if not (0 <= ra_deg <= 360 and -90 <= dec_deg <= 90):
            continue
        
        # Extract discovery date if present
        disc_date = None
        date_match = re.search(r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})', line)
        if date_match:
            disc_date = date_match.group(1)
        
        events.append({
            'name': name,
            'ra_deg': ra_deg,
            'dec_deg': dec_deg,
            'discovery_date': disc_date,
            'line_num': i,
        })
    
    # Deduplicate by name (keep first occurrence)
    seen = set()
    unique_events = []
    for e in events:
        if e['name'] not in seen:
            seen.add(e['name'])
            unique_events.append(e)
    
    return pd.DataFrame(unique_events)


def fetch_lightcurve_for_event(
    name: str,
    ra_deg: float,
    dec_deg: float,
    cache_dir: Path,
) -> tuple[Path | None, str | None, dict]:
    """Fetch lightcurve for an event by cone search, return path and ASAS-SN ID."""
    info = {'ra_deg': ra_deg, 'dec_deg': dec_deg}
    
    try:
        # Cone search to find the source
        df_cat = cone_search(ra_deg, dec_deg, radius_arcsec=5.0)
        if df_cat.empty:
            # Try larger radius
            df_cat = cone_search(ra_deg, dec_deg, radius_arcsec=15.0)
        
        if df_cat.empty:
            return None, None, {**info, 'error': 'no_source_in_cone'}
        
        # Get the closest source
        if 'asas_sn_id' in df_cat.columns:
            asas_sn_id = str(df_cat.iloc[0]['asas_sn_id'])
        elif 'id' in df_cat.columns:
            asas_sn_id = str(df_cat.iloc[0]['id'])
        else:
            return None, None, {**info, 'error': 'no_id_column'}
        
        # Download lightcurve
        lc_path, meta = download_lightcurve_by_id(asas_sn_id, cache_dir=cache_dir)
        
        return lc_path, asas_sn_id, {**info, 'asas_sn_id': asas_sn_id, **meta}
        
    except Exception as e:
        return None, None, {**info, 'error': str(e)}


def run_recovery_test(
    events_df: pd.DataFrame,
    output_dir: Path,
    *,
    max_events: int | None = None,
    fit_workers: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the microlensing pipeline on known events and measure recovery."""
    from scripts.microlensing import (
        fit_candidate_context,
        _prepare_lightcurve_df,
    )
    
    cache_dir = SKYPATROL_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if max_events is not None:
        events_df = events_df.head(max_events)
    
    total = len(events_df)
    
    results = []
    
    # Use tqdm only if not verbose (verbose prints its own output)
    iterator = events_df.iterrows()
    if not verbose:
        iterator = tqdm(iterator, total=total, desc='Testing events')
    
    for i, (idx, row) in enumerate(iterator, start=1):
        name = row['name']
        ra_deg = row['ra_deg']
        dec_deg = row['dec_deg']
        
        result = {
            'event_name': name,
            'ra_deg': ra_deg,
            'dec_deg': dec_deg,
            'discovery_date': row.get('discovery_date'),
            'recovered': False,
            'fit_ok': False,
            'quality_tier': None,
            'quality_score': np.nan,
            'best_model': None,
            'tE_days': np.nan,
            'u0': np.nan,
            't0_jd': np.nan,
            'reduced_chi2': np.nan,
            'delta_bic_vs_flat': np.nan,
            'n_points': 0,
            'error': None,
        }
        
        try:
            # Fetch lightcurve
            lc_path, asas_sn_id, fetch_info = fetch_lightcurve_for_event(
                name, ra_deg, dec_deg, cache_dir
            )
            
            if lc_path is None:
                result['error'] = fetch_info.get('error', 'fetch_failed')
                results.append(result)
                continue
            
            result['asas_sn_id'] = asas_sn_id
            
            # Prepare lightcurve DataFrame
            try:
                df_lc, band_label = _prepare_lightcurve_df(lc_path, prefer_g_band=True)
            except Exception as e:
                result['error'] = f'lc_prep_failed: {e}'
                results.append(result)
                continue
            
            if df_lc.empty or len(df_lc) < 20:
                result['error'] = f'insufficient_points: {len(df_lc)}'
                results.append(result)
                continue
            
            result['n_points'] = len(df_lc)
            
            # Build context and run fit
            context = {
                'candidate_id': name,
                'asas_sn_id': asas_sn_id or name,
                'row': {},
                'payload': {
                    'candidate_id': name,
                    'ra_deg': ra_deg,
                    'dec_deg': dec_deg,
                },
                'lc_path': lc_path,
                'df': df_lc,
                'band_label': band_label,
            }
            
            fit_result = fit_candidate_context(context)
            
            if fit_result is None:
                result['error'] = 'fit_returned_none'
                results.append(result)
                continue
            
            summary = fit_result.get('summary', {})
            
            # Extract results
            result['fit_ok'] = bool(summary.get('fit_ok', False))
            result['best_model'] = summary.get('best_model')
            result['quality_tier'] = summary.get('quality_tier')
            result['quality_score'] = float(summary.get('quality_score', np.nan))
            result['tE_days'] = float(summary.get('raw_paczynski_tE_days', np.nan))
            result['u0'] = float(summary.get('raw_paczynski_u0', np.nan))
            result['t0_jd'] = float(summary.get('fit_t0_jd', np.nan))
            result['reduced_chi2'] = float(summary.get('paczynski_reduced_chi2', np.nan))
            result['delta_bic_vs_flat'] = float(summary.get('delta_bic_vs_flat', np.nan))
            result['n_points_fit'] = int(summary.get('n_points_fit', 0))
            
            # Determine if "recovered"
            # Criteria: fit_ok=True, best_model is paczynski, and reasonable chi2
            is_paczynski = result['best_model'] == 'paczynski'
            good_fit = result['fit_ok'] and result['reduced_chi2'] < 10.0
            result['recovered'] = is_paczynski and good_fit
            
        except Exception as e:
            result['error'] = f'exception: {e}'
            traceback.print_exc()
        
        results.append(result)
        
        # Verbose output
        if verbose:
            _print_event_result(i, total, result)
    
    return pd.DataFrame(results)


def _print_event_result(i: int, total: int, result: dict) -> None:
    """Print a single event result to terminal."""
    name = result['event_name']
    width = len(str(total))
    prefix = f"[{i:>{width}}/{total}]"
    
    if result.get('error'):
        # Error case
        err = result['error']
        if len(err) > 40:
            err = err[:37] + '...'
        print(f"{prefix} {name}: \033[33m⚠ error: {err}\033[0m")
    elif result.get('recovered'):
        # Recovered successfully
        tier = result.get('quality_tier', '?')
        tE = result.get('tE_days', float('nan'))
        chi2 = result.get('reduced_chi2', float('nan'))
        tier_color = {'Gold': '33', 'Silver': '37', 'Bronze': '33', 'Suspect': '31'}.get(tier, '0')
        print(f"{prefix} {name}: \033[32m✓ recovered\033[0m "
              f"(\033[{tier_color}m{tier}\033[0m, tE={tE:.1f}d, χ²={chi2:.2f})")
    else:
        # Not recovered
        model = result.get('best_model', '?')
        chi2 = result.get('reduced_chi2', float('nan'))
        fit_ok = result.get('fit_ok', False)
        status = 'fit_ok' if fit_ok else 'fit_failed'
        print(f"{prefix} {name}: \033[31m✗ not recovered\033[0m "
              f"(model={model}, {status}, χ²={chi2:.2f})")


def generate_summary(results_df: pd.DataFrame) -> str:
    """Generate a human-readable summary of recovery results."""
    total = len(results_df)
    recovered = results_df['recovered'].sum()
    fit_ok = results_df['fit_ok'].sum()
    
    lines = [
        "=" * 60,
        "MICROLENSING RECOVERY TEST SUMMARY",
        "=" * 60,
        f"Date: {datetime.now().isoformat()}",
        f"Total events tested: {total}",
        "",
        "RECOVERY RATES:",
        f"  Recovered (Paczynski fit OK): {recovered}/{total} ({100*recovered/total:.1f}%)",
        f"  Any fit OK: {fit_ok}/{total} ({100*fit_ok/total:.1f}%)",
        "",
    ]
    
    # Quality tier breakdown
    if 'quality_tier' in results_df.columns:
        tier_counts = results_df[results_df['recovered']]['quality_tier'].value_counts()
        lines.append("QUALITY TIER DISTRIBUTION (recovered events):")
        for tier in ['Gold', 'Silver', 'Bronze', 'Suspect']:
            count = tier_counts.get(tier, 0)
            if count > 0:
                lines.append(f"  {tier}: {count} ({100*count/recovered:.1f}%)")
        lines.append("")
    
    # Best model breakdown
    if 'best_model' in results_df.columns:
        model_counts = results_df['best_model'].value_counts()
        lines.append("BEST MODEL DISTRIBUTION:")
        for model, count in model_counts.items():
            lines.append(f"  {model}: {count} ({100*count/total:.1f}%)")
        lines.append("")
    
    # Error breakdown
    error_df = results_df[results_df['error'].notna()]
    if len(error_df) > 0:
        lines.append(f"ERRORS: {len(error_df)} events")
        error_counts = error_df['error'].str.split(':').str[0].value_counts()
        for err, count in error_counts.head(5).items():
            lines.append(f"  {err}: {count}")
        lines.append("")
    
    # Statistics for recovered events
    recovered_df = results_df[results_df['recovered']]
    if len(recovered_df) > 0:
        lines.append("STATISTICS (recovered events):")
        lines.append(f"  tE median: {recovered_df['tE_days'].median():.1f} days")
        lines.append(f"  tE range: [{recovered_df['tE_days'].min():.1f}, {recovered_df['tE_days'].max():.1f}] days")
        lines.append(f"  u0 median: {recovered_df['u0'].median():.3f}")
        lines.append(f"  χ² median: {recovered_df['reduced_chi2'].median():.2f}")
        lines.append(f"  Quality score median: {recovered_df['quality_score'].median():.3f}")
        lines.append("")
    
    # Non-recovered events
    not_recovered = results_df[~results_df['recovered'] & results_df['error'].isna()]
    if len(not_recovered) > 0:
        lines.append(f"NOT RECOVERED (no error): {len(not_recovered)} events")
        lines.append("  Sample events:")
        for _, row in not_recovered.head(5).iterrows():
            lines.append(f"    {row['event_name']}: model={row['best_model']}, χ²={row['reduced_chi2']:.2f}")
        lines.append("")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Test microlensing recovery rate')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Maximum number of events to test')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--output', type=str, default='output/recovery_test',
                        help='Output directory')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to known events CSV (default: input/asas_sn_microlens.csv)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print each event result as it runs')
    args = parser.parse_args()
    
    # Paths
    if args.input:
        events_csv = Path(args.input)
    else:
        events_csv = REPO_ROOT / 'input' / 'asas_sn_microlens.csv'
    
    if not events_csv.exists():
        print(f"Error: Events CSV not found: {events_csv}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse events
    print(f"Parsing events from {events_csv}...")
    events_df = parse_asassn_microlens_csv(events_csv)
    print(f"Found {len(events_df)} events with valid coordinates")
    
    if events_df.empty:
        print("No events to test!")
        sys.exit(1)
    
    # Run recovery test
    print(f"\nRunning recovery test (max_events={args.max_events}, verbose={args.verbose})...")
    results_df = run_recovery_test(
        events_df,
        output_dir,
        max_events=args.max_events,
        fit_workers=args.workers,
        verbose=args.verbose,
    )
    
    # Save results
    results_path = output_dir / 'recovery_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Generate and save summary
    summary = generate_summary(results_df)
    print(summary)
    
    summary_path = output_dir / 'recovery_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")
    
    # Exit code based on recovery rate
    recovery_rate = results_df['recovered'].mean()
    if recovery_rate < 0.5:
        print(f"\nWARNING: Low recovery rate ({recovery_rate:.1%})")
        sys.exit(1)
    
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
