from pathlib import Path

try:
    # Running from repo root: `python calder/script.py`
    from df_process_naive import read_df_csv_naive
except ImportError:
    # Running with project root on sys.path: `python -m calder.script`
    from calder.df_process_naive import read_df_csv_naive


REQUIRE_BOTH = False

# Resolve to the peak_results directory next to this script, robust to CWD
ROOT = Path(__file__).resolve().parent / "peak_results"
CSV_FILES = [
    "peaks_12_12_5.csv",
    "peaks_12_5_13.csv",
    "peaks_13_13_5.csv",
    "peaks_14_14_5.csv",
]


def main():
    for name in CSV_FILES:
        src = ROOT / name
        if not src.exists():
            print(f"[skip] Missing: {src}")
            continue

        df_sel = read_df_csv_naive(
            src,
            require_both=REQUIRE_BOTH,
            out_csv_path=None,  # -> <stem>_selected_dippers.csv in same dir
            write_csv=True,
            index=False,
            # Also generate PDF plots for selected rows
            plot=True,
            plot_band="g",
            plots_root=ROOT,
            peak_option=False,
        )
        # Also generate V-band plots (no rewrite of CSV)
        _ = read_df_csv_naive(
            src,
            require_both=REQUIRE_BOTH,
            out_csv_path=None,
            write_csv=False,
            index=False,
            plot=True,
            plot_band="v",
            plots_root=ROOT,
            peak_option=False,
        )
        dest = src.parent / f"{src.stem}_selected_dippers.csv"
        print(f"[ok] {name}: selected {len(df_sel)} rows -> {dest}")


if __name__ == "__main__":
    main()
