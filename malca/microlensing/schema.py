"""Column names and versioning for joint microlensing products."""

MICROLENSING_JOINT_VERSION = "joint-pspl-v2"

# Compact candidate-level summary optionally materialized in Review. Detailed
# per-survey results stay in the three Parquet science products.
MICROLENSING_JOINT_COLUMN_SPECS: tuple[tuple[str, str, str], ...] = (
    ("microlensing_joint_version", "TEXT", "text"),
    ("microlensing_joint_status", "TEXT", "select"),
    ("microlensing_joint_n_surveys", "INTEGER", "float"),
    ("microlensing_joint_n_datasets", "INTEGER", "float"),
    ("microlensing_joint_t0_jd", "REAL", "float"),
    ("microlensing_joint_u0", "REAL", "float"),
    ("microlensing_joint_tE_days", "REAL", "float"),
    ("microlensing_joint_reduced_chi2", "REAL", "float"),
    ("microlensing_joint_delta_bic_flat", "REAL", "float"),
    ("microlensing_joint_parallax_attempted", "INTEGER", "bool"),
    ("microlensing_joint_parallax_preferred", "INTEGER", "bool"),
    ("microlensing_joint_parallax_delta_bic", "REAL", "float"),
    ("microlensing_joint_piE_N", "REAL", "float"),
    ("microlensing_joint_piE_E", "REAL", "float"),
)
