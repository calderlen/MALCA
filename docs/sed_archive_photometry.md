# Archive-backed SED photometry

MALCA separates catalog acquisition, archive coverage, image measurement, and
model eligibility. A nearest catalog source or a downloaded image is not, by
itself, a validated SED point.

## Acquisition

Run catalog queries and product discovery together:

```bash
conda run -n malca python -m malca sed-photometry \
  output/runs/<run>/review/review.db \
  --candidate-id <candidate_id> \
  --sources all \
  --review-db output/runs/<run>/review/review.db
```

The primary adapters are:

- AllWISE: IRSA `allwise_p3as_psd`
- Spitzer catalog: IRSA `slphotdr4`
- Spitzer images: IRSA SIA collection `spitzer_seip`
- Herschel catalogs: IRSA HPPSC2 plus HSA SPIRE point-source tables
- Herschel products: HSA observation/product retrieval
- LABOCA 870 micron catalog: ESO `ATLASGAL_V1`
- Other APEX bolometer observations: ESO TAP/DataLink discovery

AllWISE payload columns and the former VizieR Spitzer route are not acquisition
fallbacks. Cache reuse requires the same catalog release, adapter version,
match policy, coordinate epoch, quality-policy version, and candidate
astrometry hash.

The command writes:

- `sed_fetch_manifest.parquet`
- `sed_archive_coverage.parquet`
- `sed_archive_products.parquet`
- `sed_image_measurement_jobs.parquet`

When a review DB is supplied, the same records are upserted into the versioned
SQLite ledgers.

## Resumable image measurement

Process a bounded number of queued jobs:

```bash
conda run -n malca python -m malca sed-image-photometry \
  output/runs/<run>/review/review.db \
  --candidate-id <candidate_id> \
  --cache-dir output/runs/<run>/cache/sed_archive \
  --max-jobs 25
```

The worker verifies WCS and target coverage, estimates a local robust
background, and emits either a provisional aperture flux or a local 3-sigma
limit. Results remain `pending_validation` and `diagnostic_only`. Instrument
aperture corrections, source confusion, and counterpart association still
require review.

ATLASGAL is the automated APEX path. Other `APEXBOL` records are discovered for
all selected targets, but stay `reduction_required`; raw LABOCA/SABOCA data are
never passed through the ordinary FITS aperture worker.

## R24 validation gate

Explicitly accept or reject measurement IDs and export the allowed inputs:

```bash
conda run -n malca python -m malca sed-r24-inputs \
  output/runs/<run>/review/review.db \
  --accept <measurement_id> \
  --validator <name> \
  --validation-version manual-r24-v1
```

Validation decisions are immutable and versioned. The latest decision controls
eligibility. Only rows whose latest decision is `accepted`/`validated` with
`r24_eligible=1` are exported. This command creates the strict R24 handoff; it
does not install an R24 model archive or claim that a model comparison has
already been run.
