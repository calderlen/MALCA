- **model**
    - "The uniform grid is used by default. For dips, the grid typically covers high event fractions (p ~ 0.9–1.0), assuming most data points are at baseline and only a small fraction dip. For jumps, the grid covers low event fractions (p ~ 0–0.5)."
        - Shouldn't the jump model be antisymmetric, so an event fraction of e.g., 0.0-0.1?
        - also should the dip/jump fraction be slightly for the LTV model to better accomodate those solutions? while the search for microlensing and circumstellar dust dimmiung should be relatively short timescales? maybe with a fraction motivated by median length of light curves in the dataset (calculated with one of the notebooks) and median dip length/jump length computed from present results (after running review on all of the current candidates)
    - Should the observed magnitude grid spread be a bit more encompassing than $p95 + 0.5 \times \text{spread}$ (for dips) and $p5 - 0.5 \times \text{spread}$ (for jumps)  of mags throughout the light curve? think about this.
    - consider a more roubst solution for individual mag errors beyond those included in dat2's. talk to chris about this
    - the $logBF$ values seem exorbitant., so $\log{BF}=\log{\text{evidence mixture}} - \log{\text{evidence baseline}}$ may be wrong for some reason. i suspect that it has to do with the magnitude grids
        - as a further investigation, compare local logBF and global logBF values for the dippers and jumpers
    - check if logBF local or logBF global are being used at each point that a logBF is used in the pipeline and review stages. also check for where LOO posterior prob vs logBF_global vs logBF_local is used and see if an alternative wouldnt be more suitable anywhere
    - "MALCA supports two triggering modes for identifying significant events:" -- ensure that LOO are the default everywhere and even consider removing the logBF_local thresholds. clearly the logBF_local threshold is wired to be a trigger in some cases
    - figure out why light curves are passing through the whole pipeline despite there being no detected peak. beacuse i keep getting light curves in the pipeline that do not have a peak and so it makes me suspect that these plots were not generated with the pass all filters enabvled? or prehpase the pass all filters condition does not check for a peak? in some cases it seems that there are indeed dimming candidates in light curves without that detected peak so i don tjust want to throw away those candidates but at least  figure out why that is happening
    - "The morphology classification uses Bayesian Information Criterion (BIC) for model selection and computes symmetry scores to quantify the ingress/egress asymmetry characteristic of dipper disk occultations (Tzanidakis+2025)."
        - are the BIC and symmetry scores for detected dips/jumps collected in the appropriate data product and thus displayed in the info panel in the gui? why or why not?
    - perhaps consider more morphology classifications than (skew) gaussian for dips and gaussian/pacyznski/fred for jumps so that this model can be extended to other things?
        - other morphologies that may be good: (1) exoplanet transits which are U-shaped or V-shaped. these have 4 or 5 parameters. (2) eclipsing binaries, 5 parameters. (3) pulsating variables, 4 parameters ? (4) supernova morphologies???, like maybe collect light curves from known supernovae, grouped by type, and either get morphologies for each type or train a ML model on them and establish types of your own? see if there isnt an existing solution to this problem. idk enough about SN light curves right now
        - another change to the code could be to fit to things that you dont want, e.g., eclipsing binaries, then just reject those light curves with best fit being an eclipsing binary instead of say a gaussian dimming event. but the thing is since the runs module preceding this works that way it does i suspsect e.g., the eclipsing binary morpology would not be adequately captured. so need to look into run gating logic for fiting to other morphologies to make sense.
        - but then maybe need to consider a removal and/or alteration of baselines as well. because to track something with a e.g., SHO-ish behavior the SHO baseline will track it and thus we will lose key information. so either would need to say do a simple median baseline, or see if a baseline subtraction is even in order for this scenario.
        - also could use it to look at variables with a strong baseline, fit to that, both characterizing the variables (perhaps) and detecting deviations from nominal physical behavior of the variables' light curve with a well-fit baseline. could be interesting need to look more into this
    - maybe the padded segment around each run that the morphology functions are fit to should be wider, so e.g. to get better tails on a gaussian fit? maybe make a notebook that compares the morphology fits to excursions to REAL sample light curves to compare fit performance across padding width, different ways of making initial guesses
        - make sure your gasussian for the DIPS and the JUMPS are oriented the right way -> this may explain the negative jump scores? you shouldnt be trying to fit a positive guasisan to a negative dip (i think). also ensure that code is aware of astronomy magntiude sign convention here so that code makes sense
        - add a comparison of a gaussian vs skew gaussian fit with corresponding BICs to see how skew a skew gausian must be for it to be selected instead of a regular gaussian
        - consider adding skew gaussian to jumps?
    - investigate the relationships between the building, filtering, and morphology fitting to runs. have a mental map of how these three operations are intertwined

- **manifest building**
    - `iter_source_records` is an abstruse name. at the very least write words out in full and use lc instead of source. also why "records"?
    - `build_manifest_dataframe` can probably be shortened to build_manifest

- **tagging**
    - the tag stage now reflects that this step mainly annotates failures before events.py
    - maybe some of the catalog crossmatches done in the reivew/chacterization stage would be more suitably done here. but i think mostly not because these are expensive (?) however maybe just biting the bullet and doing a bulk lookup (where possible) for a bunch of these canddiates and storing the catalog locally would be a lot better than doing a lookup each time in the later stages. think more about this
    - i think at the very least the crossmatch to gaia dr3 variability, asas-sn catalog, ztf catalog, and atlas catalog would be more sensibly done here instead of downstream. this information would be useful beyond just this narrow pipeline.
    - chunk size should be increased from 5000 to 100000

- **filtering**
    - should add a MAX run count to `filter_run_robustness` in addition to the minimum run count. this may be a cheap way to filter out periodic candidates
    - filtering order
        `filter_posterior_strength` -> `filter_run_robustness` -> `filter_morphology` -> 

- **validation**
    - validation filters are disabled by default? 
    - why are the filter and validation modules separate?

- **data output**
    - generating plots with malca stv-plot before review may just be a waste of space. at the very least add a toggle on/off for plot generation at this stage since the LC info is bundled anyway and looked at in the reviews stage. default behavior should be NOT to plot at this stage.
    - 

- **review**
    - get a notebook running that profiles all of the light curves i will see in a review queue for a certain mag bin BEFORE i run the review. i would like to see population statistics on what sort of light curves i am looking at before i spend hours running a review. $\frac{30000}{\frac{60}{5}\cdot60} = 41.7$ means that i will need to spend 40 hours in a single week just to get through 30k light curves while working at a pace of 5 seconds per light curve. 
    - add a pace/timer to the review gui next to the review count
    - phase-folded light curve, external spectra, and even external light curves (?) should be displayed in the review module alongside the ASAS-SN light curves. check all of the data objects that are pulled externally and see which can be visua

- **LTVs**
    - need to work on this. it seems like an easier problem over all but need to run the code on the deepeber mag bins with a lower threshold for long term variability, e.g. <0.3 mag. the extant 2-component gaussian mixture model will not work for these, so i need to construct a different bayesian model to complete this task.
    - also consider just splitting off this code into a separate repository and renaming MALCA to something more focused on these one-off events rather than long-term variability. because for now lumping them otgether doesnt make sense since their core modules are distinct. but maybe it does because a lot of stuff downstream from the model can be used for the LTVs as well, especially the review module.
