# Dipper literature catalog (version 2026-07-16)

This is a versioned, provenance-preserving working catalog for crossmatching a systematic ASAS-SN dipper search. It is not a claim that the astronomical literature is closed or that every object below has the same physical origin.

## Contents

- `dipper_literature_master.csv`: one row per spatially deduplicated source/system.
- `dipper_literature_occurrences.csv`: one row per coordinate-bearing paper/object occurrence, retaining classifications, quality flags, source URL, and notes.
- `dipper_coordinates.txt`: plain-text master coordinate list.
- `dipper_paper_retrieval_audit.csv`: paper-by-paper access and retrieval outcome, including supplied-screen-shot membership.
- `dipper_paper_retrieval_summary.txt`: plain-text grouping of every audited paper by retrieval category.
- `dipper_literature_catalog.xlsx`: the same tables in a filterable workbook.

The nine supplied VizieR records are data-catalog entries attached to papers, not additional papers. They were used where useful but are not double-counted in the paper audit.

## Release counts

- 8,778 unique paper/object occurrences from 60 coordinate-bearing papers.
- 8,031 spatially deduplicated sources/systems.
- 7,929 masters have at least one explicit dipper label; 7,753 have at least one YSO-oriented dipper classification.
- 8,024 masters are recommended for the broad crossmatch; rejected/nonmember-only rows are retained but flagged.
- 96 screened papers or literature records in the audit.

Evidence-tier counts in the master table:

- `author_vetted_or_explicit`: 1,327
- `automated_catalog`: 239
- `automated_primary`: 3,537
- `automated_secondary_or_tentative`: 2,863
- `broad_analogue_or_inventory`: 58
- `rejected_or_nonmember`: 7

Retrieval-category counts in the audit:

- `all_candidate_information_retrieved`: 60
- `partial_candidate_information_retrieved`: 4
- `candidate_information_not_retrieved`: 6
- `paper_not_accessible`: 0
- `not_attempted`: 26

## Scope and flags

The supplied Zotero folder mixes classical low-mass YSO dippers with main-sequence dusty dippers, circumbinary/circumsecondary eclipses, dipping giants, exocomet-like events, and single long obscurations. Those populations are retained, but `population_tiers`, `core_yso_dipper_flag`, `explicit_dipper_label_flag`, and `evidence_tier` keep them separable.

Mas et al. (2026) dominates the catalog. Its 6,542 Gaia candidates are not all equivalent: 3,708 are primary/high-quality and 2,834 are secondary; the paper warns that the secondary QPD sample is contamination-prone. Cody et al. (2025) similarly has 13 confident and 27 candidate-only Herbig Ae/Be dipper targets. Rebull et al. (2022) contributes 46 members/probable members and 10 rejected/nonmember Dip-flag rows. These distinctions are retained.

The 44 EWOCS-VI Westerlund 1 rows are called periodic dippers and were visually confirmed, but the authors propose binary eclipses and report no disk colors. They therefore use the `periodic_dimming_binary_candidate` population tier.

## Coordinate and deduplication method

Duplicate extracts of the same paper were collapsed only across input datasets, using shared identifiers or 1.25 arcsec. The master table then links occurrences from different papers by strong identifiers (up to 30 arcsec, to tolerate epoch/proper-motion differences) or by a 2.0 arcsec position match. A guard prevents two distinct rows from the same paper from entering one master group. The coordinate from the highest-priority occurrence is retained, favoring Gaia DR3, then Gaia DR2 and machine-readable survey coordinates.

Coordinates are decimal degrees and reflect the frame/epoch supplied by each publication, typically ICRS/J2000-style catalog positions. No proper-motion propagation was performed. For ASAS-SN, use this list as a candidate crossmatch and then verify the Gaia/2MASS counterpart in blended fields; do not rely on positional proximity alone.

## Known incomplete or unrecoverable sets

- **Catalogue of UBVRI photometry of T Tauri stars and analysis of the causes of their variability (1994)** — `candidate_information_not_retrieved`. Do not label the full T Tauri catalog as dippers.
- **YSOVAR: Mid-infrared Variability in the Star-forming Region Lynds 1688 (2014)** — `candidate_information_not_retrieved`. Paper was accessible but adopts a different morphology scheme and does not provide a transferable explicit dipper list.
- **Near-infrared Variability in the rho Ophiuchi Molecular Cloud (2014)** — `candidate_information_not_retrieved`. Screened through later citations; no modern paper-level dipper membership list was recovered in this pass.
- **Near-infrared Variability in the Orion Nebula Cluster (2015)** — `partial_candidate_information_retrieved`. The published VizieR catalog omits the AA-Tau/dipper membership flag. Coordinates are recovered for 10 explicitly plotted periodic examples and all 6 explicitly labelled irregular examples; do not treat as a complete list.
- **Dipper stars in the Upper Sco and rho Oph star forming regions identified from K2 (2017)** — `candidate_information_not_retrieved`. Full conference paper retrieved. It reports 24 new dippers, but neither a target table nor EPIC/2MASS identifiers are published; Figure 1 uses only internal numbers 1-26. Coordinates cannot be reconstructed from this publication alone.
- **A survey for variable young stars with small telescopes: First results from HOYS-CAPS (2018)** — `partial_candidate_information_retrieved`. The paper reports 101 dipper table rows using unrounded M>0.5. The public table has 96 definite rows, including five exact duplicate entries, leaving 91 definite unique sources. Nine additional unique sources print M=0.50; exactly five of the nine satisfy the unrounded cut, but their identities cannot be recovered. All nine are included and explicitly flagged so an exhaustive crossmatch does not miss one.
- **Infrared variability of young solar analogues in the Lagoon Nebula (2022)** — `partial_candidate_information_retrieved`. Article and source retrieved. Candidate membership is in machine-readable supporting Table A1 (classification QPD/APD); the supporting file endpoint was not accessible in this pass. No coordinates are asserted from incomplete excerpts.; Do not use 15 as the total dipper count; it is only the WTTS subset.
- **Time-series Photometry and Multiwavelength Characterization of the Young Stellar Population in Mon R2 (2023)** — `candidate_information_not_retrieved`. Paper accessible; it reports 10 multi-event and 4 single-event eclipse/occultation light curves in appendix figures, but no clean machine-readable dipper flag/table was recovered.
- **A survey for variable young stars with small telescopes - VIII. Properties of 1687 Gaia selected members in 21 nearby clusters (2024)** — `candidate_information_not_retrieved`. Important omission. Class boundary is acknowledged as arbitrary; contact authors/use database for a reproducible row list.
- **Unveiling the structural content of NGC 6357 via kinematics and NIR variability (2024)** — `partial_candidate_information_retrieved`. Paper gives subgroup percentages, not a reliable total catalog count; filter the electronic table.

No screened paper was classified as wholly inaccessible: when the publisher page was unavailable, an arXiv, CDS/VizieR, repository, or bibliographic route was found. Some electronic supporting tables nevertheless could not be retrieved, as listed above.

## Completeness statement

The supplied screenshots are not exhaustive. Major omitted coordinate catalogs found in this search include Mas et al. (2026; 6,542 Gaia candidates), Zhang et al. (2024; 296), Rebull et al. (2018/2020/2022; 83/21/56), Venuti et al. (2021; 17), McGinnis et al. (2015; 33), Cody, Hillenbrand & Rebull (2022; 24), Cody et al. (2025; 40), Tajiri et al. (2020; 35), Bodman et al. (2017; full 25-object Upper Sco/rho Oph set), and Ordenes-Huanca et al. (2026; 44 Westerlund 1 periodic dippers).

This release is exhaustive only with respect to the screened discovery/catalog papers and accessible tables through the cutoff date. Single-object follow-up and theory literature is open-ended; representative missing follow-ups are recorded in the audit, but the absence of a follow-up paper does not imply the source is absent from the coordinate master.

## Recommended use

1. Begin with `recommended_crossmatch_flag=yes` for a broad veto/reference list.
2. For a classical YSO-only comparison, additionally require `core_yso_dipper_flag=yes` and inspect `evidence_tier`.
3. Keep `automated_secondary_or_tentative`, `broad_analogue_or_inventory`, and `periodic_dimming_binary_candidate` rows separate in statistics.
4. Cite the originating paper(s) from the occurrence table, not this compilation alone.
