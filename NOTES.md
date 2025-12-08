arjuna:     249.89 GiB of RAM and 80 processors
draupadi:   249.89 GiB of RAM and 80 processors
nankipoo:   250.02 GiB of RAM and 80 processors
sahadeva:   249.89 GiB of RAM and 80 processors
titipu:     250.03 GiB of RAM and 80 processors


12_12.5 takes ~
12.5_13 takes ~
13_13.5 takes ~ 2.75 hrs to run
13.5_14 takes ~ 7.5 hrs to run ?
14_14.5 takes ~ 6.5 hrs to run
14.5_15 takes ~



ASAS-SN/VSX+BJ: matched 10 / 31 within 2.0"
                           sum  count
category                             
Dippers                      6     11
Multiple Eclipse Binaries    2      7
Single Eclipse Binaries      2     13

ASASSN+BJ: matched 31 / 31 within 2.0"
                           sum  count
category                             
Dippers                     11     11
Multiple Eclipse Binaries    7      7
Single Eclipse Binaries     13     13

VSX-cleaned+BJ: matched 4 / 31 within 2.0" --> this may be because you corect for proper motion only after creating vsx-cleaned, which would explain why ASAS-SN/VSX+BJ has more candidates because that uses the pm-corrected vsx dataset
                           sum  count
category                             
Dippers                      3     11
Multiple Eclipse Binaries    0      7
Single Eclipse Binaries      1     13

VSX-raw+BJ: matched 17 / 31 within 2.0"
                           sum  count
category                             
Dippers                      9     11
Multiple Eclipse Binaries    2      7
Single Eclipse Binaries      6     13





- ok so we are switching to significance based detection relative to a (global or per-camera?) GP baseline
    - decide what significance you're going to require?
        - from the literature, something physically motivated?
        - 0.3 mag threshold seems arbitrary











- Future improvements identified; these approaches shoudl allow searches with lower thresholds on delta g across a wider range of apparent magnitudes -- BJ's paper
    - Automate the idenitification of stars so that fewer sources require manual inspection
    - Filter out known variables with well-defined types, e.g., EBs
    - Improve methods to filter typical stellar variability; present dispersion limit can eliminate longer, deeper events ebcause they are large enough to affect the dispersion while also not eliminating high amplitude variable stars that trigger our simple approach to flagging candidates
    - better automate the identification of bad data points and remove them before beginning the search

- timescale
    - SP: used seasonal medians to eliminate shorter time scale variaility, then linear and quadratic fits to the medians were used to identify sources slowly varying in brightness
    - BJ: only required one point in light curve to exceed the magnitude threshold, along with surviving a bunch of other cuts, to identify a candidate because it was more focused on magnitude changes than timescale

- Dippers
    - before doing anything else, write code to
        - automate identification of stars so that fewer sources require manual inspection
        - filter out known variables with well-defined types
        - improve methods to filter typical stellar variability
        - find a way to decrease the present dispresion limit because it can eliminate longer, deeper events because they are large enough to affect the dispersion while also not elimiating high ampltiude varialbe stars that the simple approach to finding candidates of delta g > 0.3 mag
        - better automate the identification of bad data points that that this can also be automated and decrease the amount of time spent on visual inspection

- Long-term variability

- Eliminating false positives
    - LTVs
    - Dippers
        - Stars affected by nearby saturated stars, you can better automate this search - done?
        - Higher amplitude varaible stars - done?
        - Automate flagging of known varaibles with well-defined types - done?


        - Improve method to filter out stellar variability beyond a simple light curve dispersion limit sigma > 0.15., because this can eliminate longer, deeper events because they are large enough to affect dispersion while not eliminating high amplitude variable stars that trigger the simple to flagging candidates
            - one improvement would be to look for upward and downward dips about the median of the light curve, since variable stars would show both brightening and dimming features
            - dipper targets would only show dimming events

        - False positives that were only eliminated by visual inspection that you can instead eliminate automatically
            - bright nearby star
                -The points scatter inconsistently, with some segments shifted up or down.
                - Different camera colors don’t agree.
            - bad point cluster
                - Almost perfectly flat baseline, except for a few vertical groups of outliers.
                - Variability comes from just a handful of bad data points.
                - A real dipper would show dips spread across many epochs, not a cluster of bad points.
            - camera offset
                - Clear step-like jumps in brightness that line up with changes in observing camera (colors).
                - Each camera measures a slightly different mean level.
                - Not true variability, but a calibration mismatch between instruments.
            - R Coronae Borealis variable
                - Multiple large, irregular fadings of >1 mag.
                - Characteristic of R CrB stars, which fade when carbon dust clouds obscure the star.
                - Real astrophysical variability, but belongs to a known variable class, not the dipper search target.
            - contact binary
                - When phase-folded, shows smooth, repeating, symmetric dips every half day.
                - Classic W UMa–type binary signature.
                - Too regular and periodic to be a one-off dipper event.
            - semi-regular variable
                - Large, smooth rises and falls in brightness over long timescales.
                - Cyclic but irregular — typical of pulsating giants.
                - Not a sharp, isolated dip but long quasi-periodic changes.
            - T Tauri type YSO
                - Noisy, irregular scatter around a sinusoidal baseline.
                - Light curve is messy but still periodic when folded.
                - Behavior matches spot modulation or accretion variability of a young star.
                - Too continuous and chaotic for the clean discrete dips of a “big dipper.”  
        - False positives, altogether
            - Instrumental/systematic false positives: 
                - bright star contamination
                - bad point clusters
                - camera offsets
            - Astrophysical but wrong class: 
                - R CrB
                - contact binaries
                - semi-regular pulsators
                - YSOs
            - True dippers would instead show isolated, deep, discrete dips with otherwise flat baselines, seen consistently across cameras.
        - Key Traits of Real Dippers (Figure 8)
            - Discrete, Deep Events
            - The dips are sharp and localized in time.
            - They drop by >0.3 mag and then return to baseline.
                - Example: J070519+061219 shows a single large drop, then recovers.
            - Flat or Stable Baseline Outside Events: Most of the light curves are essentially flat except during the dip events. Contrast with false positives, where the whole light curve may look noisy, offset, or quasi-periodic.
                - Example: J081816–430955 has a clean baseline before and after two distinct dips.
            - Consistency Across Cameras
                - Different colors (ASAS-SN cameras) track the same event.
                - The dip is seen in all cameras simultaneously, not just one (unlike camera offsets or bad-point clusters).
            - Event Morphology
                - Dips can be U-shaped or V-shaped, sometimes asymmetric.
                - Multiple dips (e.g., J081523–385923) look like repeated discrete obscurations, not smooth sinusoidal variability.
                - In contrast, binaries (false positives) show perfectly repeating patterns.
                - Duration is Intermediate
                - Not a fast contact binary dip (hours).
                - Not a multi-year monotonic trend (like R CrB or AGB variables).
                - Instead: dips last days–weeks, then recover.
                - Example: J183153–284827 fades steeply over ~year and stays low for a time — still discrete compared to a decade-long slow fade.





- need to add stellar flare filters