"""Microlensing analysis settings: fitting, morphology, quality scoring, contamination checks."""

# =============================================================================
# FLUX-SPACE FITTING
# =============================================================================

# Reference magnitude for flux conversion (None = use median of LC)
FLUX_REF_MAG_DEFAULT = None

# Minimum relative flux (avoids log(0) issues)
FLUX_MIN_RELATIVE = 1e-12

# Blending parameter bounds
BLEND_FS_MIN = 0.0  # Source flux lower bound (can be 0 for high blending)
BLEND_FB_MIN = 0.0  # Blend flux lower bound

# =============================================================================
# MODEL FITTING PARAMETERS
# =============================================================================

# Paczynski model bounds
PACZYNSKI_U0_MIN = 1e-3
PACZYNSKI_U0_MAX = 2.0
PACZYNSKI_TE_MIN_DAYS = 0.5
PACZYNSKI_TE_MAX_FACTOR = 4.0  # Max tE = factor * LC span

# FRED (nova) model bounds
FRED_TAU_MIN_DAYS = 0.2
FRED_TAU_MAX_FACTOR = 1.0  # Max tau = factor * LC span

# Gaussian model bounds
GAUSSIAN_SIGMA_MIN_DAYS = 0.2
GAUSSIAN_SIGMA_MAX_FACTOR = 1.0  # Max sigma = factor * LC span

# Fitting optimization
FIT_MAX_NFEV = 5000  # Max function evaluations for least_squares
FIT_SOFT_L1_SCALE = 1.5  # f_scale for soft_l1 loss
FIT_MULTISTART_U0 = [0.02, 0.2, 1.0]  # u0 starting values
FIT_MULTISTART_TE_FRACTIONS = [0.08, 0.18, 0.35]  # tE as fraction of LC span

# =============================================================================
# MODEL SELECTION (BIC-based)
# =============================================================================

# ΔBIC thresholds (BIC_other - BIC_paczynski)
DELTA_BIC_POSITIVE = 2.0  # Weak evidence for Paczynski
DELTA_BIC_STRONG = 6.0  # Strong evidence for Paczynski
DELTA_BIC_VERY_STRONG = 10.0  # Very strong evidence

# Selection thresholds
NON_PACZYNSKI_SELECTION_THRESHOLD = -6.0  # Alt model preferred if ΔBIC <= this
PAC_WEAK_VS_FLAT_MIN_DELTA_BIC = 6.0  # Min ΔBIC for Pac to beat flat
PLOT_ALT_WHEN_PAC_STRUGGLES = -10.0  # Plot alts if flat beats Pac by this much

# =============================================================================
# MORPHOLOGY METRICS
# =============================================================================

# Rise/decay time measurement
MORPH_HALF_MAX_THRESHOLD = 0.5  # Fraction of peak for half-max time
MORPH_QUARTER_MAX_THRESHOLD = 0.25  # Fraction for quarter-max time

# Event window definition (in units of tE)
MORPH_EVENT_WINDOW_TE_FACTOR = 2.0  # Event window = ±factor * tE

# Outside-event analysis
MORPH_OUTSIDE_WINDOW_TE_FACTOR = 3.0  # Outside = beyond ±factor * tE
MORPH_EXCURSION_SIGMA_THRESHOLD = 3.0  # Sigma threshold for excursions

# Symmetry score (uses trapezoid integration)
MORPH_SYMMETRY_MIN_POINTS = 5  # Min points per side for symmetry calc

# Residual autocorrelation
MORPH_RESIDUAL_AUTOCORR_MAX_LAG = 10  # Max lag for autocorr analysis

# =============================================================================
# CV/NOVA TEMPLATE FITTING
# =============================================================================

# Nova (FRED) classification
NOVA_TAU_RISE_MAX_DAYS = 10.0  # Fast rise characteristic
NOVA_TAU_DECAY_MIN_DAYS = 5.0  # Slow decay characteristic
NOVA_AMPLITUDE_MIN_MAG = 0.5  # Minimum amplitude for nova-like

# CV classification thresholds (from classify.py, replicated for clarity)
CV_BP_RP_BLUE_THRESHOLD = 0.5  # Blue excess
CV_G_ABS_FAINT_THRESHOLD = 8.0  # Faint absolute magnitude
CV_CONTAMINATION_BASE_PROB = 0.3

# Secondary peak detection
SECONDARY_PEAK_MIN_SEPARATION_DAYS = 30.0  # Min separation between peaks
SECONDARY_PEAK_MIN_AMPLITUDE_FRAC = 0.3  # Min amplitude as fraction of primary
SECONDARY_PEAK_MAX_COUNT = 5  # Max secondary peaks to report

# =============================================================================
# PERIODICITY SCANNING
# =============================================================================

# Period search range
PERIOD_MIN_DAYS = 0.1
PERIOD_MAX_DAYS = 500.0
PERIOD_N_GRID = 10000

# Significance thresholds
PERIOD_LSP_FAP_THRESHOLD = 0.01  # FAP < this = significant
PERIOD_PDM_THETA_THRESHOLD = 0.5  # Theta < this = significant
PERIOD_CE_ENTROPY_THRESHOLD = 2.0  # Entropy < this = significant

# Alias periods to flag
PERIOD_ALIAS_PERIODS_DAYS = [1.0, 0.5, 29.53, 365.25, 182.625]
PERIOD_ALIAS_TOLERANCE_FRAC = 0.05  # Relative tolerance for alias matching

# Residual periodicity (after Paczynski subtraction)
RESIDUAL_PERIOD_POWER_THRESHOLD = 0.3  # LSP power threshold for residuals

# =============================================================================
# CROWDING / BLENDING
# =============================================================================

# Gaia neighbor search
CROWDING_SEARCH_RADIUS_ARCSEC = 5.0  # Search radius for neighbors
CROWDING_BRIGHT_DELTA_MAG = 3.0  # Neighbors brighter than target + this

# Crowding thresholds
CROWDING_MAX_COUNT = 10  # Max neighbors before flagging
CROWDING_MAX_BRIGHT_COUNT = 3  # Max bright neighbors before flagging

# Blend flux estimation
BLEND_PSF_FWHM_ARCSEC = 2.5  # Typical PSF FWHM for ASAS-SN
BLEND_CONTAMINATION_THRESHOLD = 0.1  # Flag if blend_flux > 10% of total

# =============================================================================
# GAIA CMD
# =============================================================================

# CMD axis ranges
CMD_BP_RP_MIN = -0.5
CMD_BP_RP_MAX = 4.0
CMD_MG_MIN = -5.0
CMD_MG_MAX = 15.0

# Extinction correction (from config_ltv.py, replicated)
CMD_R_V = 3.1
CMD_A_G_PER_AV = 0.789
CMD_E_BP_RP_PER_AV = 0.415

# Background sample
CMD_BACKGROUND_SAMPLE_SIZE = 10000  # Max background points to plot
CMD_BACKGROUND_ALPHA = 0.1  # Transparency for background

# =============================================================================
# PARALLAX FITTING
# =============================================================================

# Parallax eligibility
PARALLAX_MIN_TE_DAYS = 80.0  # Only fit if tE >= this
PARALLAX_MIN_FIT_POINTS = 80  # Min points for parallax fit
PARALLAX_MIN_SPAN_DAYS = 240.0  # Min LC span for parallax

# Parallax parameter bounds
PARALLAX_MAX_ABS_PIE = 1.5  # Max |πE| (Earth-Sun scale)
PARALLAX_MAX_U0_ABS = 2.0  # Max |u0|
PARALLAX_U0_FACTOR_MIN = 1.0 / 3.0  # Min u0 factor relative to PSPL
PARALLAX_U0_FACTOR_MAX = 3.0  # Max u0 factor relative to PSPL
PARALLAX_TE_FACTOR_MIN = 0.35  # Min tE factor relative to PSPL
PARALLAX_TE_FACTOR_MAX = 3.0  # Max tE factor relative to PSPL

# Parallax selection
PARALLAX_REQUIRED_DELTA_BIC = 6.0  # ΔBIC for parallax to be preferred
PARALLAX_MAX_REDUCED_CHI2 = 10.0  # Max reduced χ² for valid fit

# MCMC settings
PARALLAX_ENABLE_MCMC = True
PARALLAX_MCMC_CHAINS = 6
PARALLAX_MCMC_BURN = 200
PARALLAX_MCMC_STEPS = 400
PARALLAX_MCMC_THIN = 2
PARALLAX_MCMC_MIN_ACCEPTANCE = 0.002
PARALLAX_RANDOM_SEED = 20260322

# =============================================================================
# QUALITY SCORING
# =============================================================================

# Component weights (must sum to 1.0)
QUALITY_WEIGHT_FIT = 0.25  # Fit quality (χ², shoulders, BIC)
QUALITY_WEIGHT_MORPHOLOGY = 0.20  # Morphology metrics
QUALITY_WEIGHT_ASTROPHYSICAL = 0.15  # CMD position, extinction, RUWE
QUALITY_WEIGHT_CONTAMINATION = 0.20  # CV score, periodicity, crowding
QUALITY_WEIGHT_PARALLAX = 0.10  # Parallax convergence
QUALITY_WEIGHT_COVERAGE = 0.10  # LC span, cadence, n_points

# Fit quality sub-scores
QUALITY_FIT_CHI2_GOOD = 2.0  # Reduced χ² below this = full score
QUALITY_FIT_CHI2_BAD = 10.0  # Reduced χ² above this = zero score
QUALITY_FIT_MIN_SHOULDERS = 3  # Min shoulder points per side
QUALITY_FIT_MIN_STRONG_POINTS = 2  # Min points above 50% depth

# Morphology sub-scores
QUALITY_MORPH_RISE_DECAY_MAX_RATIO = 3.0  # Max asymmetry ratio
QUALITY_MORPH_SKEWNESS_MAX = 1.5  # Max acceptable skewness
QUALITY_MORPH_AUTOCORR_MAX = 0.5  # Max residual autocorrelation
QUALITY_MORPH_SYMMETRY_MAX = 0.5  # Max symmetry score deviation

# Coverage sub-scores
QUALITY_COVERAGE_MIN_POINTS = 50  # Min points for full score
QUALITY_COVERAGE_MIN_SPAN_DAYS = 365.0  # Min span for full score
QUALITY_COVERAGE_MIN_CAMERAS = 2  # Min cameras for bonus

# Quality tier thresholds
QUALITY_TIER_GOLD = 0.8  # Score >= this = Gold
QUALITY_TIER_SILVER = 0.6  # Score >= this = Silver
QUALITY_TIER_BRONZE = 0.4  # Score >= this = Bronze
# Below Bronze = Suspect

# =============================================================================
# CANDIDATE GRID PLOT
# =============================================================================

# Grid layout
GRID_MAX_COLS = 4  # Max columns in grid
GRID_PANEL_WIDTH = 3.5  # Inches per panel width
GRID_PANEL_HEIGHT = 2.5  # Inches per panel height
GRID_DPI = 300  # Output DPI

# Panel content
GRID_ZOOM_TE_FACTOR = 3.5  # Zoom window = ±factor * tE
GRID_MIN_ZOOM_DAYS = 35.0  # Minimum zoom half-window

# Selection criteria
GRID_MIN_QUALITY_TIER = "Silver"  # Minimum tier for inclusion
GRID_MAX_CANDIDATES = 25  # Maximum candidates in grid

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# PDF output
PLOT_DPI = 300
PLOT_FORMAT = "pdf"

# Column prefixes for output
OUTPUT_PREFIX_MORPHOLOGY = "morph_"
OUTPUT_PREFIX_PERIODICITY = "period_"
OUTPUT_PREFIX_CROWDING = "crowd_"
OUTPUT_PREFIX_QUALITY = "quality_"
