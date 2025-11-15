from reproduce_candidates import build_reproduction_report

report = build_reproduction_report(
    out_dir="./results_test",   
    out_format="csv",                 
    n_workers=10,                    
)

print(report[["source","source_id","mag_bin",
              "detected","detection_details",
              "g_n_peaks","v_n_peaks","matches_expected"]])