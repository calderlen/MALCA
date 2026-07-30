# SFR membership data

`sfr_association_crosswalk.csv` is the curated bridge between MALCA's
distance-resolved molecular-cloud regions and association hypotheses returned
in `banyan_probabilities_json`. Only rows with
`include_in_sfr_probability=true` contribute to the mapped-SFR probability.
The mapping is intentionally conservative: a similarly named association is
not linked when its physical distance or population does not represent the
cloud used by the Mollweide analysis.

`sfr_catalog_members.csv` is the normalized exact-membership table consumed by
`malca.enrichment.sfr_membership`. It currently contains its schema but no
published member rows. This makes catalog evidence explicitly unavailable
instead of silently interpreting absence from an unpopulated table as
non-membership.

To add a published catalog, append one row per accepted stellar member and
retain:

- the Gaia source identifier as text;
- the catalog name and bibliographic reference;
- the association and mapped SFR names;
- the catalog's membership probability or quality flag, when supplied;
- Gaia parallax and proper-motion values plus uncertainties/correlations, when
  available, so the same catalog can define the local astrometric comparison
  model.

Set `accepted_member=false` for catalog entries that are retained for audit but
must not count as exact membership or train the local kinematic model. If an
older catalog uses Gaia DR2 identifiers, cross-match those identifiers to DR3
before adding them and record that provenance in `catalog_reference` or
`catalog_quality`.

An empty catalog therefore yields `sfr_catalog_match_status=catalog_unavailable`.
It never yields `not_listed`, and it does not prevent mapped BANYAN or
cloud-environment evidence from being evaluated independently.
