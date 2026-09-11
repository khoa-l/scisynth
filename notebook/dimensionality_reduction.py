import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Problem statement

    Analysts build intuition for messy scientific data slowly, through years of exposure. There is no fast, controllable way to practice recognizing the sampling effects, instrument artifacts, and distributional quirks that make real data hard to interpret. If you wanted to see examples of the many effects, distributions, and transformations you would need to program it yourself. There is no tool that parametrizes the data generation.'

    Separately, people who develop dimensionality-reduction (DR) methods lack a clean way to ask "what happens to *this specific* distortion when I project the data," because real datasets come without a known ground truth to compare against.

    This tool lets an analyst encode the known factors of their real data into a spec and generate a matched surrogate whose ground truth is known, so candidate DR techniques can be judged on the surrogate and the choice carried back to the real data More generally, this tool generates synthetic datasets with a fully known generating process, applies a configurable stack of realistic corruptions to them, and (optionally) runs any DR technique on the result while keeping the provenance links across all three stages. It does not perform analysis. It produces data and can be used in sense-making and intuition building.

    The design goal is that the tool is **domain-agnostic**: its parameters describe the *structure* and *defects* of data rather than the specific science. It should be able to stand in for a chemistry spectrum, an astronomy light curve, or a soil survey, time-series geological data through combinations of the same primitives, with named presets that approximate specific fields.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Conceptual model and shared vocabulary
    The tool is organized around a three-layer pipeline.

    | Layer | What it is |
    |---|---|
    | **Latent** (`L`) | The generated structure: the manifold/process and its intrinsic parameters. This is the "ground truth". This is what our user-study would compare the task performance to.
    | **Observed** (`O`) | `L` after the **observation model**: a stacked, ordered set of sampling and distortion operators. This is what an analyst is handed and treats as "the data." |
    | **Projected** (`P`) | `O` after a **projection** (a DR technique). The low-dimensional embedding. |

    Two operators connect them:

    - **Observation model** (`L -> O`): an ordered pipeline of *sampling operators* (how much of `L` is measured, and where) and *distortion operators* (how each measurement is degraded). The operations will be composable and independently seeded.
    - **Projection** (`O -> P`): any DR map, supplied by the user or pulled from an existing library. The tool doesn't implements DR itself.

    Every element carries **provenance**: each row/cell in `O` knows which latent point it came from (each observed data point may correspond to one **or** multiple points / many-to-many relationship / "low-resolution camera analogy") and which operators touched it, and each point in `P` knows its `O` and `L` origin. This is what lets the tool answer "where did this outlier in the embedding come from, and was it real signal or an injected artifact." This will be the **provenance trace**.

    Keep in mind for the research that when a observations happen, the observations would live on a manifold that isn't in the same space a the latent space, but instead a joint function of the latent distributions & observation model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Goals

    - Represent a wide span of EDA input structures (sequential and non-sequential, temporal and static, spatial and non-spatial, small-n and large-n, low and high intrinsic dimension) from a small set of composable primitives.
    - Parametrize the latent structure and the distortions **independently**, so any distortion can be applied to any structure.
    - Ground the available structures and distortions in established taxonomies rather than ad-hoc choices (see references).
    - Run any user-supplied or library DR technique on `O`, and expose standard DR quality metrics against the known `L`.
    - Preserve a full provenance trace across `L -> O -> P`.
    - Support a blinded workflow: `O` looks real; `L` and the spec are retrievable only through an explicit call that is logged.
    - Ship as a pip-installable package usable in Jupyter, functions first, optional widgets on top.

    ## 4. Non-goals

    - Not an EDA assistant. No auto-profiling, no suggested plots, no analysis of user data.
    - Not a DR implementation. So it would orchestrate DR, but does not reimplement PCA/UMAP/etc.
    - Not first-principles scientific simulation. The bar is statistical and structural realism, not physical accuracy.
    - Not a study-delivery harness (timers, consent, condition assignment), but it will be used in the user-study.
    - No answer grading. `L` is exposed for both the study and for users to compare with; any answer comparison logic will need to be separately created for the study.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Users and primary use cases

    1. **EDA analyst (calibration):** sweeps distortion parameters on a fixed `L` to learn what, for example, specific noises or block missingness looks like across different data shapes.
    2. **EDA analyst (DR selection):** encodes the known factors of their own real dataset (dimensionality, cluster count, noise level, missingness pattern, etc.) into a spec, generates a matched surrogate where the ground truth is known, runs several candidate DR techniques on the surrogate to see which ones faithfully recovers that known structure, and carries that technique back to their real data. Also, allows the the analyst to explore _how_ different techniques affect their data.
    3. **EDA analyst (diagnosis practice):** receives `O` with `L` withheld, forms hypotheses about the generating process and defects, then reveals `L` to self-check.
    4. **DR developer:** generates `O` with known `L`, projects with a candidate method, and inspects `P` plus quality metrics and the provenance trace to see how the method handles specific corruptions and manifold geometries.
    5. **Researcher/instructor:** generates matched dataset sets (same spec varying seed, or one parameter swept) for study conditions, and exports usage logs.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Layer L: latent structure generators

    `L` is built by choosing a **generative family** and its parameters. Families are grouped along orthogonal structural axes so the space is described by combinations, not an enumeration of domains.

    ### 6.1 Structural axes

    | Axis | Values |
    |---|---|
    | Ordering | non-sequential (i.i.d. rows) / sequential (ordered, e.g. time or genome position) |
    | Temporal behavior | static / evolving (drift, trend, regime change) |
    | Spatial embedding | non-spatial / spatial (point cloud or lattice, 1D–3D) / spatiotemporal |
    | Sample size | small-n (<30, common in wet-lab work) to large-n (10^6+, catalogs and grids) |
    | Ambient vs. intrinsic dimension | `p` observed features over an intrinsic dimension `d <= p` |

    The manifold hypothesis (Fefferman, Mitter & Narayanan 2016) motivates separating ambient dimension `p` from intrinsic dimension `d`: real high-dimensional scientific data typically concentrates near a low-dimensional manifold, and recovering `d` is exactly what DR and intrinsic-dimension estimators attempt do.

    ### 6.2 Generative families (proposed v1 set)

    | Family | Description | Key params | Grounding |
    |---|---|---|---|
    | Independent features | mixed continuous/categorical columns, specified marginals | per-column distribution, cardinality | baseline tabular |
    | Correlated / latent-factor | features generated from shared latent factors with a covariance/loading structure | # factors, loadings, noise | factor models; realistic correlation |
    | Structural causal model (SCM) | features generated along a DAG of mechanisms | graph, mechanism types | SCM priors used in TabPFN (Hollmann et al. 2023); gives realistic conditional structure and lets missingness/anomalies be defined causally |
    | Cluster mixtures | Gaussian/non-Gaussian mixtures, adjustable separation and balance | k, separation, class balance | canonical EDA/clustering benchmark |
    | Manifolds | swiss roll, S-curve, sphere, torus, helix, Möbius strip, embedded low-d subspace | intrinsic `d`, ambient `p`, curvature, sampling density | standard manifold-learning benchmarks (Tenenbaum et al. 2000; scikit-learn generators; QuIIEst-style ID benchmarks) |
    | Time series | trend + seasonality + autocorrelated noise; changepoints; multi-channel | length, cadence, AR/seasonal params, changepoints | time-series decomposition; anomaly benchmarks (Lai et al. 2021) |
    | Spatial fields | Gaussian random fields with tunable spatial autocorrelation on points or lattices | variogram range/sill/nugget, extent, resolution | geostatistics; Tobler's First Law (1970); Cressie 1993 |
    | Gridded/image | 2D/3D fields or stacks (spatiotemporal) | shape, bands, feature scale | remote sensing, microscopy, model output |
    | Counts/compositional | Poisson/negative-binomial counts, zero-inflation, compositional (parts of a whole) | rate, dispersion, sparsity | sequencing, mineralogy, ecology |
    | Event/survival | time-to-event with a hazard model and censoring | hazard, censoring type/rate | reliability, clinical follow-up |
    | Graph/network | nodes and edges from a topology model | n, edge model, community structure | interaction networks, phylogenies |

    A dataset can compose families (for example, a spatial field of cluster labels sampled through time), so "spatiotemporal multi-channel with latent clusters" is a combination of these different factors.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Layer O: the observation model (distortion and sampling catalog)

    Operators are grouped by the taxonomies below. Each is an independent operator with its own strength parameter and seed, applied in an explicit order so effects stack realistically (calibration drift *then* saturation behaves differently from the reverse). The catalog is anchored in three papers: missing-data mechanisms (Rubin 1976; Little & Rubin), data-quality/dirty-data taxonomies (Rahm & Do 2000; Kandel et al.; Wickham; Hellerstein), and anomaly typologies (Chandola, Banerjee & Kumar 2009; Lai et al. 2021).

    Some of these factors I understand better than others, but nothing that I couldn't figure out how to implement with some time from my look at the papers.

    ### 7.1 Sampling operators (which of `L` is measured)

    | Operator | Effect | Grounding |
    |---|---|---|
    | Density / subsampling | overall n; uniform down to sparse | small-n vs large-n realism |
    | Non-uniform sampling | oversample/undersample regions of the manifold or domain (e.g. Beta-weighted along a coordinate) | distorts perceived manifold density (EntroPath 2026 shows this changes DR/geodesic recovery) |
    | Irregular temporal sampling | uneven time spacing, gaps, downtime | sensor reality; drives aliasing |
    | Clustered spatial sampling | preferential sampling near accessible sites | geostatistical sampling bias |
    | Selection / survivorship bias | inclusion probability depends on a value (e.g. only bright/large objects observed) | truncation bias, a common astronomy/ecology effect |
    | Unequal replication | uneven group/site sample sizes | unbalanced designs |

    ### 7.2 Missingness operators

    Rubin's three mechanisms, plus structured variants:

    | Operator | Mechanism |
    |---|---|
    | MCAR | missingness independent of all values |
    | MAR | missingness depends on other observed values |
    | MNAR | missingness depends on the missing value itself (e.g. below detection limit) |
    | Block / interval missing | contiguous gaps (instrument downtime) |
    | Censoring | left/right/interval (values known only to be beyond a bound) |

    ### 7.3 Noise and instrument operators

    | Operator | Effect |
    |---|---|
    | Additive Gaussian | homoscedastic noise, set by SNR |
    | Poisson / shot | count-dependent noise |
    | Heteroscedastic | noise scales with signal or with a covariate |
    | Multiplicative | proportional error |
    | Colored (1/f, thermal) | non-white noise spectrum |
    | Calibration drift / offset | slow baseline change or fixed bias |
    | Saturation / clipping | ceiling or floor on the sensor |
    | Quantization | discretization to instrument resolution |
    | Limit of detection | floor below which values are unreliable or censored |
    | Batch / site effects | systematic per-instrument or per-site offsets |

    ### 7.4 Outlier and anomaly operators

    Following Chandola et al. (2009), with time-series types from Lai et al. (2021):

    | Operator | Type |
    |---|---|
    | Point outliers | isolated extreme values (adjustable contamination fraction) |
    | Contextual anomalies | normal in isolation, abnormal in context (broken seasonality, spatial context) |
    | Collective anomalies | a subsequence/region abnormal as a group (stuck sensor, trend/shapelet/seasonal segment) |

    ### 7.5 Structural, temporal, and integrity operators (out-of-scope)

    | Operator | Effect | Grounding |
    |---|---|---|
    | Autocorrelation | injected serial or spatial dependence | violates i.i.d. assumptions |
    | Non-stationarity | trend, variance change, regime shift | common in climate/sensor data |
    | Changepoints | abrupt shifts in level or dynamics | changepoint literature |
    | Clock/timezone errors | misaligned or duplicated timestamps | Gschwandtner et al. dirty-time-data category |
    | Duplicates / near-duplicates | repeated or fuzzy-repeated records | Rahm & Do instance-level issues |
    | Unit / encoding errors | mixed units, inconsistent categorical labels, mixed dtypes in a column | Kandel et al.; Hellerstein entry/integration errors |
    | Tidy-structure violations | values in headers, multiple variables per column | Wickham messy-data problems |
    | Dimensionality issues | p >> n, multicollinearity, class imbalance | high-dimensional EDA pitfalls |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Different features will have different spatial dependencies. Imagine we have two kinds of crystals and they have fluorescence with near-infrared, blue, etc. Our measurement takes a radius of the point, but the point might have a different distribution depending on other data.

    Our original latent should represent the thing that the analyst / scientist is trying to understand. We would want a distribution over latent points and then per-sample have some influence per-channel on the sample. If we say that the long form is the "canonical" form, then the weighted bi-partite graph would work to display the data in the provenance. Have the wide data as an option.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8. Layer P: projection (pluggable DR) and evaluation

    The tool orchestrates DR but does not implement it. Also, a decision between re-implementing the DR metrics versus using an existing tool. Here are the suggestion for the interfaces:

    1. **Library adapter:** anything exposing scikit-learn's `fit_transform` (PCA, MDS, Isomap, LLE, Laplacian eigenmaps, t-SNE, UMAP, PHATE, PaCMAP, TriMap, autoencoders via a wrapper).
    2. **Callable adapter:** a user function `f(X) -> Y` for a method under development. This is the primary path for DR developers.
    3. **Metric adapter:** metrics are delegated to an existing evaluation library, primarily ZADU (Jeon et al. 2023), which already implements the standard reliability measures.

    The DR-evaluation literature systematizes metrics into local, cluster-level, and global classes (Bertini et al. 2011; Espadoto et al. 2019; Thrun et al. 2023). Supported set:

    - **Local:** trustworthiness and continuity (Venna & Kaski), mean relative rank error, LCMC, co-ranking-matrix summaries.
    - **Cluster-level:** label separation / silhouette against known latent clusters.
    - **Global:** distance or rank correlation between `L` and `P`; residual variance; reconstruction error where applicable.
    - **Manifold recovery:** correlation of recovered coordinates against the true manifold parameters when `L` is a known manifold (e.g. geodesic vs. embedded distance).

    Support distinguishing between the changes in our defects / distortion parameters versus the changes to the DR hyperparameters?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Layer V: visualization
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9. Provenance trace ("mapping distortions through the pipeline")

    There is a substantial line of work on visualizing and interpreting DR distortions (Aupetit 2007; Stahnke et al. 2016; Cavallo & Demiralp 2017; Nonato & Aupetit 2019; Bian et al. 2020; Chatzimparmpas et al. 2020; Sankaran et al. 2026, the `distortions` package). Almost all of it must estimate distortion from the embedding itself, because with real data the true structure is unknown. With a known L, projection distortions are computed exactly rather than inferred, and each can be attributed to a specific observation-model operator.

    We adopt the distortion vocabulary from that literature (Nonato & Aupetit 2019): **missing neighbors** (points close in L pulled apart in P), **false neighbors** (points distant in L placed together, the source of spurious clusters), and **tears/compression/stretch** of the manifold. The provenance trace would hopefully make these queryable and operator-attributable:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 9.1 Provenance is a weighted many-to-many lineage, not an id

    Our original naive design attached one `latent_id` to each observed row, assuming the observation model is a bijection (a one-to-one pairing of items in two sets). Real data often changes the _number_ of records in the examples listed below:

    - **Fan-in (many latent to one observed):** binning/histogramming, spatial pixel or areal aggregation (the [modifiable areal unit problem](https://en.wikipedia.org/wiki/Modifiable_areal_unit_problem)), temporal downsampling to an average, deduplication or record linkage that merges distinct entities.
    - **Fan-out (one latent to many observed):** repeated measurements and replicates, resampling or bootstrap, interpolation or regridding onto a finer grid, a duplicate-record operator.
    - **No latent origin:** injected point outliers, imputed values, and interpolation artifacts are observed points with no true L ancestor.
    - **No observed descendant:** rows removed by row-level sampling or dropout.

    To consider for this, we'll make provenance a provenance graph / sparse weighted bipartite relation `W` of shape `(n_O, n_L)`, where `W[o, l]` is the contribution of latent point `l` to observed point `o`.

    This would mean that a 1:1 inherited point has a single weight of 1 whereas an aggregated point has several weights summing to 1. In the other cases, a replicated latent point appears in several observed rows; a synthetic point has an all-zero row; a dropped latent point has an all-zero column.

    In regards to standard DR techniques, most map `O` to `P` row-for-row, so `P` inherits `W` directly (landmark or coreset methods that are not 1:1 compose a second, analogous matrix). For the ones that don't (landmark, coreset), we would compose a seocnd matrix.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 9.2 Additions to the tracked information

    The lineage model requires additions to the data carried at each layer:

    - **Per observed point:** origin_kind in {inherited, aggregated, replicated, synthetic, imputed}, n_sources (count of latent contributors), and the weight row of W.
    - **Per latent point:** n_descendants and a dropped flag (from the column of W).
    - **Per operator:** a tag to let us know how the operator affected the data?, one of {`preserves_rows`, `adds_rows`, `removes_rows`, `merges_rows`}. Any operator that changes the cardinality must return an updated lineage, not just updated values, so that `W` can be created correctly down the pipeline.

    We need to figure out we want to handle flags for things like `is_boolean`. I guess we can do a partial number or maybe get rid of it all together. Do we also want to distinguish between the metrics we run comparing `O` to `L` (how faithful the data is to the tru / latent vs. observed) verus the metrics run between `P` and `O` (how faithful the data is against the input)? Would we also want `O` and `P` metrics? Is one of those in scope versus not for the other? If we do run `O` and `L` metrics, and there is not a bijection in the data, then how do we consider for that?

    Would we re-map `O` back into the `L` space? Would we not compare certain metrics for certain types of data / comparison groups (L/O, L/P, O/P)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 10. Domain presets

    We could have domain presets that are named bundles of `L` family + observation-model parameters that approximate a field's characteristic data, so a user can start from something familiar without the tool being "about" that field specifically.

    Examples, non-exhaustive:
    - `spectroscopy_like` (peaks + baseline drift + heteroscedastic noise)
    - `light_curve_like` (periodic signal + irregular cadence + gaps + point outliers)
    - `soil_survey_like` (spatial field + clustered sampling + detection limit)
    - `single_cell_like` (high-p counts + zero-inflation/dropout + batch effects)
    - `climate_station_like` (trend + seasonality + autocorrelation + block missing)

    Find domain persons to help build the preset dataset for the tool.

    Represent presets as are just parameter dictionaries.

    Two of these map onto fields where DR interpretation is actively contested and therefore make good training targets: single-cell genomics (Chari & Pachter 2021; Lause et al. 2024; Bombina et al. 2025) and chemical-space visualization (Orlov et al. 2024). A cross-domain critical analysis of how DR is (mis)used (Cashman et al. 2025) motivates keeping the preset set diverse rather than tuned to one field.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 11. Data model and API sketch

    This is an example of how the tool might work. Not a final spec sheet and the functiosn, etc. can all be changed if something doesn't seem right. I still need to explore `scikit` and other tools to understand the specific implementation details.

    ```python
    from scisynth import generate, reveal, observe, project, metrics

    # L: pick a latent family
    latent = generate.manifold("swiss_roll", n=2000, ambient_dim=3,
                               sampling="beta(1,4)", seed=7)

    # O: apply an ordered observation model
    obs = observe(latent, pipeline=[
        {"op": "heteroscedastic_noise", "snr": 12},
        {"op": "mnar_missing", "threshold": "low", "rate": 0.1},
        {"op": "point_outliers", "fraction": 0.02},
    ], seed=7)

    df = obs.data                      # what the analyst sees
    truth = reveal(obs.id)             # latent params + operator flags

    # P: run any DR technique, get metrics against known L
    import umap
    emb = project(obs, umap.UMAP().fit_transform)
    q = metrics(emb)                   # trustworthiness, continuity, co-ranking, ...
    emb.color_by("operators_applied") # provenance-aware plotting
    ```

    - Functions are the primary API. A `Spec` dataclass (fully serializable) could support batch/programmatic generation for the study harness.
    - Optional `manifest.widgets.explore("...")` gives ipywidgets sliders over `L` and observation-model params with live plots. I think widgets would be really cool for ease of use & visualizations?
    - `reveal()` would be the only way to see `L`; make sure that `obs.data`/`.metadata` never leak latent values or operator identities.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 12. Non-functional requirements

    - **Packaging:** pip-installable; core deps numpy/scipy/pandas; optional extras for xarray (gridded), networkx (graph), ipywidgets (widgets), and DR libraries.
    - **Reproducibility:** seeded throughout, independent per-operator seeds, and reliable outputs based on seed and input parameters. Both our tool and the libraries used.
    - **Performance:** up to ~10^6 rows/cells in a few seconds on a laptop? Probably not as important for the MVP, but something to worry about considering the scale of the types of datasets some scientific workflows deal with.
    - **Validity tests:** internal tests confirming generated data matches requested parameters within tolerance (e.g. measured SNR, missingness rate, intrinsic dimension).
    - **Extensibility:** new latent family, operator, or DR adapter. One function against a documented interface for extending the base capability with no need for core changes
    - **Docs:** parameter ranges and units in docstrings; one gallery notebook per structural axis and per operator showing `L`, `O`, and a sample `P`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 13. Out of scope for v1

    - Realistic instrument file formats (FITS/NetCDF/messy CSV with junk headers/encodings). Valuable for realism, large surface area, but probably not something we need to worry about. We can probably preface that the tool doesn't account for this currently, but could be something to think about synthesizing? (something to talk to a domain expert with)
    - Overlay/comparison against held-out real datasets. For visualization and comparison purposes? I'm not exactly sure how it would look, but I just imagine that it would be good to have some way to overlay the synthetially generated data over existing ones.
    - More presets to represent possibly the degree of corruptions, more domains?
    - Non-notebook or non-Python delivery. I think it'd be cool to have this as a visualization tool just on the web.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## References (the rest is included in the Semantic Scholar link)
    - Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley.
    - Rubin, D. B. (1976). Inference and missing data. *Biometrika* 63(3), 581–592.
    - Little, R. J. A. & Rubin, D. B. *Statistical Analysis with Missing Data.*
    - Rahm, E. & Do, H. H. (2000). Data cleaning: problems and current approaches. *IEEE Data Eng. Bull.*
    - Kandel, S., Paepcke, A., Hellerstein, J. M. & Heer, J. (2012). Enterprise data analysis and visualization: an interview study. *IEEE TVCG* 18(12), 2917–2926. (Data-quality anomaly categories: missing, erroneous, inconsistent, extreme, key violations.)
    - Wickham, H. (2014). Tidy data. *J. Stat. Softw.*
    - Hellerstein, J. M. (2008). *Quantitative Data Cleaning for Large Databases.* UNECE. (Entry/measurement/distillation/integration error sources.)
    - Chandola, V., Banerjee, A. & Kumar, V. (2009). Anomaly detection: a survey. *ACM Comput. Surv.* 41(3), Article 15. (point/contextual/collective)
    - Lai, K.-H. et al. (2021). Revisiting time series outlier detection: definitions and benchmarks. *NeurIPS Datasets & Benchmarks Track.* (Time-series anomaly types: global, contextual, trend, shapelet, seasonal.)
    - Fefferman, C., Mitter, S. & Narayanan, H. (2016). Testing the manifold hypothesis. *JAMS* 29(4), 983–1049.
    - Tenenbaum, J. B., de Silva, V. & Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction (Isomap; swiss-roll benchmark). *Science* 290(5500), 2319–2323.
    - van der Maaten, L., Postma, E. & van den Herik, J. (2009). Dimensionality reduction: a comparative review.
    - Lee, J. A. & Verleysen, M. (2009). Quality assessment of dimensionality reduction: rank-based criteria (co-ranking matrix). *Neurocomputing* 72(7–9), 1431–1443.
    - Venna, J. & Kaski, S. (2001). Neighborhood preservation in nonlinear projection methods: an experimental study (trustworthiness and continuity). *ICANN 2001*, 485–491.
    - Espadoto, M., Martins, R. M., Kerren, A., Hirata, N. S. T. & Telea, A. C. (2019). Toward a quantitative survey of dimension reduction techniques (local/cluster/global metric classes). *IEEE TVCG* 27(3), 2153–2173.
    - Hollmann, N., Müller, S., Eggensperger, K. & Hutter, F. (2023). TabPFN: a transformer that solves small tabular classification problems in a second (SCM prior for synthetic tabular data). *ICLR 2023.*
    - Tobler, W. (1970). First law of geography; Cressie, N. (1993). *Statistics for Spatial Data* (variogram/kriging).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Testing the `scisynth` package
    """)
    return


@app.cell
def _():
    from scisynth import LatentData, ObservedData, Provenance, ProvenanceStep, Spec, project

    from scisynth.latent import Independent, Tabular, generate, LatentDistribution
    from scisynth.latent import FAMILY_REGISTRY, register_family  # for extensions

    from scisynth.observed import UniformSubsample, GaussianNoise
    from scisynth.observed import ObservedData, observe, ObservationState
    from scisynth.observed import OPERATOR_REGISTRY, register_operator  # for extensions

    from sklearn.decomposition import PCA

    import numpy as np
    from scipy.stats import norm, poisson, randint

    return (
        GaussianNoise,
        Independent,
        PCA,
        Tabular,
        UniformSubsample,
        norm,
        np,
        observe,
        poisson,
        project,
        randint,
    )


@app.cell
def _(Independent):
    _dist = Independent(p=5, scale=1.0)
    _data = _dist.rvs(n=200, seed=42)

    mo.ui.table(_data.to_frame())

    spec = _dist.to_spec()
    mo.vstack([
        mo.md(f"Spec: `{spec}`"),
        mo.ui.table(_data.to_frame()),
    ])
    return


@app.cell
def _(Tabular, norm, np, poisson, randint):
    # independent mixed-type columns
    _dist = Tabular(columns=[norm(0, 1), poisson(mu=5), randint(0, 4)])
    _data = _dist.rvs(n=200, seed=42)

    # To test the copula path
    corr = [[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]]

    _dist_corr = Tabular(
        columns=[norm(0, 1), norm(0, 1), poisson(mu=3)], corr=corr
    )
    _data_corr = _dist_corr.rvs(n=500, seed=0)

    empirical = np.corrcoef(_data_corr.X[:, 0], _data_corr.X[:, 1])[0, 1]

    mo.vstack(
        [
            mo.md(f"Empirical correlation: {empirical:.3f}"),
            mo.ui.table(_data_corr.to_frame()),
            mo.ui.table(_data.to_frame()),
        ]
    )
    return


@app.cell
def _(GaussianNoise, Independent, PCA, UniformSubsample, observe, project):
    _latent = Independent(p=3).rvs(n=500, seed=0)
    _latent.to_frame()

    _observed, _prov = observe(
        _latent,
        operators=[
            UniformSubsample(frac=0.4),
            GaussianNoise(sigma=0.2),
        ],
        seed=42,
    )

    [op.name for op in _observed.operators]

    for i, step in enumerate(_prov.observed_steps):
        print(
            f"step {i} | {step.operator.name} | X: {step.X.shape} | ids[:5]: {step.ids[:5]}"
        )

    _prov.observed_steps[0].layer  # "observed"
    _prov.observed_steps[0].X  # data after this step
    _prov.observed_steps[0].ids  # ids after this step
    _prov.observed_steps[0].to_frame()

    _projected, _prov = project(_observed, PCA(n_components=2), _prov, seed=42)

    mo.vstack(
        [
            mo.ui.table(_latent.to_frame()),
            mo.ui.table(_observed.to_frame()),
            mo.ui.table(_projected.to_frame()),  # id, z0, z1
            mo.ui.table(_prov.to_frame()),
            mo.ui.table(_prov.observed_steps[0].to_frame()),
            mo.ui.table(_prov.projected_steps[0].to_frame()),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    1. Generate some mixture models with clustering for our latent distribution (min 2 classes)
    2. Generate a projected scatter plot that we give to users
    3. Give some provisional interaction on the projected scatter plot
    4. Match the actions with the coherency of the user's clustering interactions
    """)
    return


if __name__ == "__main__":
    app.run()
