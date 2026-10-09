# /// script
# requires-python = ">=3.10"
# dependencies = ["scisynth[viz]", "marimo", "scikit-learn", "umap-learn"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import json
    from pprint import pp

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import scipy.stats as stats
    from plotly.subplots import make_subplots
    from sklearn.decomposition import PCA
    from umap import UMAP

    import scisynth as sy
    from scisynth.testing.invariants import check_observation, observations_equal
    from scisynth.viz import plot, plot_observations


@app.cell
def _():
    # latent through sampler produces an observation
    # observations through the stages produces a final observation
    # a generator holds config; realize(seed) draws one ground truth.
    # any parameter can be a distribution (one draw per realization).
    domain = sy.Domain.from_extents([(0, 10), (0, 10)], units=["mm", "mm"])
    generator = sy.GaussianField(
        domain, length_scale=1.5, amplitude=stats.uniform(0.5, 1.0)
    )
    latent = generator.realize(seed=1)

    # evaluate() queries anywhere: (n_points, n_channels)
    print(latent.kind, latent.evaluate(np.array([[2.0, 3.0], [5.0, 5.0]])).shape)
    return domain, generator, latent


@app.cell
def _(domain, generator, latent):
    # analytic fields, arithmetic (sum & product), and stacked channels
    ramp = sy.AnalyticField(lambda x, y: 0.1 * x, domain)
    combined = ramp + latent * latent

    # multichannel takes realized fields; corr mixes them into correlated channels
    fields = [generator.realize(seed=s) for s in (1, 2, 3)]
    corr = [[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]]
    correlated = sy.Multichannel(*fields, corr=corr)

    print(combined.evaluate(np.array([[2.0, 3.0]])), correlated.n_channels)
    return (correlated,)


@app.cell
def _(correlated, latent):
    # grid: both endpoints included. points: uniform in the domain.
    # values is always (*spatial, n_channels).
    grid_obs = sy.GridSampler((64, 48)).run(latent, seed=0)
    points_obs = sy.PointSampler(500).run(latent, seed=0)
    multi_obs = sy.GridSampler((64, 48)).run(correlated, seed=0)

    print(grid_obs.values.shape, grid_obs.layout)
    print(points_obs.values.shape, points_obs.layout)
    print(multi_obs.values.shape, multi_obs.n_channels)
    return grid_obs, multi_obs


@app.cell
def _(grid_obs):
    # grids have coords by axis name; mask True where valid
    # 'truth' on observations returned is the clean samples;
    # ids follow each location through the pipeline; meta logs every step
    print(list(grid_obs.coords), grid_obs.mask.shape, grid_obs.truth.shape)
    print(grid_obs.ids.shape, grid_obs.size)
    pp(grid_obs.meta)
    return


@app.cell
def _(latent, multi_obs):
    # Sampler.plot samples, then plots; the layout comes from the observation
    # plots are registered by 'kind' and
    # classes declare which 'kind's of plots they accept
    # multi-channel observations plot one channel at a time.
    mo.vstack(
        [
            mo.ui.plotly(sy.GridSampler((80, 40)).plot(latent)),
            mo.ui.plotly(sy.PointSampler(n=3200).plot(latent)),
            mo.ui.plotly(plot(multi_obs, "channel 2", channel=2)),
        ]
    )
    return


@app.cell
def _(grid_obs):
    # a stage runs on its own; meta records the parameters it drew
    noisy = sy.GaussianNoise(0.1).run(grid_obs, seed=0)
    print(noisy.meta[-1]["name"], np.nanstd(noisy.values - grid_obs.values))
    return


@app.cell
def _():
    # preview(): before and after for every defect run on a built-in demo signal
    stages = [
        sy.GaussianNoise(sigma=stats.uniform(0.05, 0.1)),
        sy.PoissonNoise(scale=0.5),
        sy.RandomDropout(0.1),
        sy.Quantize(levels=4),
        sy.Clip(lo=-0.5, hi=0.5),
        sy.Saturate(ceiling=0.5),
        sy.ValueOffset(0.5),
        sy.Gain(2.0),
        sy.Drift(rate=0.5, axis="x"),
        sy.CoordinateShift(0.1),
        sy.PositionJitter(0.1),
        sy.RandomInsertion(100),
        sy.Downsample(2),
    ]

    mo.vstack([stage.preview(kind="heatmap") for stage in stages])
    return


@app.cell
def _(multi_obs):
    # can preview on your own observation as well, one channel at a time
    mo.ui.plotly(sy.GaussianNoise(0.2).preview(multi_obs, channel=2))

    mo.ui.plotly(
        sy.RandomInsertion(100).preview(multi_obs, channel=2, kind="heatmap", bins=25)
    )
    return


@app.cell
def _(latent):
    # a sampler followed by stages: a simulated instrument
    observer = sy.Observer(
        sy.GridSampler(128),
        [
            sy.GaussianNoise(sigma=stats.uniform(0.05, 0.1)),
            sy.RandomDropout(p=0.02),
            sy.Quantize(levels=12),
        ],
    )
    observation = observer.run(latent, seed=42)

    # bare stages are named after their class; the sampler comes first
    print(observer.names)
    return observation, observer


@app.cell
def _(observer):
    # observer.stages is a StageList: index by position, name or slice
    _stages = observer.stages
    print(_stages[0], _stages["quantize"], _stages[:2].names, _stages.items()[0])
    return


@app.cell
def _(latent, observer):
    # rebuild from parts with with_stages; or name steps explicitly
    first_two = observer.with_stages(observer.stages[:2])
    longer = observer.with_stages([*observer.stages.items(), sy.Clip(hi=0.5)])
    named = sy.Observer(
        ("camera", sy.PointSampler(2000)),
        [("noise", sy.GaussianNoise(0.1)), sy.Quantize(8)],
    )
    print(first_two.names, longer.names)
    print(named.names, named.run(latent, seed=0).meta[-1]["name"])
    return (first_two,)


@app.cell
def _(latent, observation, observer):
    # same seed, same result; error against the truth where valid
    a, b, c = (observer.run(latent, seed=s) for s in (1, 1, 2))
    print(observations_equal(a, b), observations_equal(a, c))

    residual = np.where(
        observation.mask[..., None], observation.values - observation.truth, np.nan
    )
    print(np.nanstd(residual))  # as compared to the clean sample
    return


@app.cell
def _(latent, observer):
    # Observer.plot runs the observer and plots every step, led by the latent.
    # panels share one color scale and coordinate frame.
    mo.ui.plotly(observer.plot(latent))
    return


@app.cell
def _(latent, observer):
    # trace() keeps every step: trace[0] is the sampler, trace[i + 1] the i-th stage
    trace = observer.trace(latent, seed=42)
    print(trace.names, np.array_equal(trace[0].values, trace[0].truth))
    print(trace["random_dropout"].mask.mean(), trace[:2].names)
    return (trace,)


@app.cell
def _(first_two, latent, trace):
    # per-stage diffs; an observer of the first k stages reproduces trace[k]
    for _diff in trace.diffs:
        print(_diff.name, _diff.n_kept, _diff.n_dropped, _diff.n_moved, _diff.changed)
    print(observations_equal(first_two.run(latent, seed=42), trace[2]))
    return


@app.cell
def _(trace):
    # 3d trace plot: one layer per step and a line
    # from each point to where it goes next.
    # grey: stayed. red cross: dropped, ending in the layer of the
    # stage that dropped it.
    # orange, purple and green appear below.
    mo.ui.plotly(trace.plot())
    return


@app.cell
def _(latent):
    # orange: moved (CoordinateShift moves every point)
    # purple: merged (Downsample averages blocks).
    # each stage records how its output was built
    # from its input (sparse links); links_between composes them.
    coarse = sy.Observer(
        sy.GridSampler(32),
        [sy.RandomDropout(0.3), sy.CoordinateShift([0.5, 0.0]), sy.Downsample(4)],
    ).trace(latent, seed=1)
    print(coarse.diffs[-1].links.shape, coarse.links_between().shape)
    mo.ui.plotly(coarse.plot(max_points=80))
    return


@app.cell
def _(latent):
    # green diamonds: added
    # RandomInsertion adds spurious points with new ids & NaN truths
    # for grid samples, point defects first flatten it to points
    scatter_trace = sy.Observer(
        sy.PointSampler(300),
        [
            sy.PositionJitter(sigma=0.15),
            sy.RandomDropout(0.2),
            sy.RandomInsertion(n=40),
        ],
    ).trace(latent, seed=3)
    for _diff in scatter_trace.diffs:
        print(_diff.name, _diff.n_moved, _diff.n_dropped, _diff.n_added)
    mo.ui.plotly(scatter_trace.plot())
    return


@app.cell
def _(observer, trace):
    # a trace, or any list of observations, also plots as a panel grid
    # (trace.plot(kind="scatter") is the same thing; kind="heatmap" draws cells)
    mo.ui.plotly(plot_observations(trace[:3], observer.names[:3], kind="heatmap"))
    return


@app.cell
def _(correlated):
    # Multi-channel: an array with one entry per channel acts per channel
    sensor = sy.Observer(
        sy.GridSampler(96),
        [
            sy.Gain([1.0, 2.0, 0.5]),
            sy.GaussianNoise(sigma=[0.05, 0.1, 0.2]),
            sy.RandomDropout(0.05),  # drops whole locations, all channels
            sy.Clip(lo=-2.0, hi=[2.0, 3.0, 1.0]),
            sy.Quantize(levels=32),  # range is per channel
        ],
    )
    result = sensor.run(correlated, seed=7)
    check_observation(result)
    mo.vstack(
        [
            mo.ui.plotly(sensor.plot(correlated, channel=1)),
            mo.ui.plotly(
                plot_observations(
                    [sy.GridSampler(96).run(correlated, seed=0), result],
                    ["clean", "sensor"],
                    channel=2,
                )
            ),
        ]
    )
    return


@app.cell
def _(domain):
    # Systematic defects. More than four panels wrap into rows; ncols sets the
    # row width and show_latent=False drops the latent panel.
    systematic = sy.Observer(
        sy.GridSampler(64),
        [
            sy.ValueOffset(0.5),
            sy.Gain(1.2),
            sy.Drift(rate=0.05, axis="x"),
            sy.CoordinateShift([0.2, -0.1]),  # moves coords; truth stays put
            sy.PoissonNoise(scale=5.0),
        ],
    )
    positive = sy.AnalyticField(lambda x, y: 3 + np.sin(x) * np.cos(y), domain)
    mo.ui.plotly(systematic.plot(positive, ncols=3, show_latent=False))
    return


@app.cell
def _():
    # Observers round-trip through dicts and JSON. Callables, distributions and
    # AnalyticField cannot be serialized.
    config = sy.Observer(
        sy.GridSampler(32), [sy.GaussianNoise(0.1), sy.Quantize(16)]
    ).to_dict()
    print(sy.Observer.from_dict(json.loads(json.dumps(config))).names)
    return


@app.cell
def _(domain):
    # six channels in two correlated groups of three, so two components describe them
    _fields = [
        sy.GaussianField(domain, length_scale=2.0).realize(seed=s) for s in range(6)
    ]
    _corr = np.eye(6)
    _corr[:3, :3] = _corr[3:, 3:] = 0.9
    np.fill_diagonal(_corr, 1.0)
    sensors = sy.Multichannel(*_fields, corr=_corr)

    sensor_trace = sy.Observer(
        sy.GridSampler(48),
        [sy.GaussianNoise(0.3), sy.RandomDropout(0.1), sy.Quantize(levels=6)],
    ).trace(sensors, seed=0)
    return (sensor_trace,)


@app.cell
def _(sensor_trace):
    # axes picks what the 3-D trace plots along x and y: a string is a coordinate
    # name, an int a channel index. Here channel 0 against channel 1, so noise and
    # quantization show up as movement instead of staying put in space.
    mo.ui.plotly(sensor_trace.plot(axes=(0, 3)))
    return


@app.cell
def _(sensor_trace):
    # projection wraps any scikit-learn style estimator (fit / transform)
    # valid locations are a sample with its channels as features
    # fit on the clean step; transform embeds any observation as points with
    # coords c0, c1 (values and ids unchanged, masked locations NaN)
    # estimators own their random_state per sklearn

    pca = sy.Projection(PCA(n_components=2)).fit(sensor_trace[0])
    print(pca.estimator.explained_variance_ratio_.round(2))
    embedded = pca.transform(sensor_trace[-1])
    print(embedded.layout, list(embedded.coords), embedded.values.shape)
    return (pca,)


@app.cell
def _(pca, sensor_trace):
    # the embedding as a scatter next to the samples it came from, all coloured by
    # channel 0 on one shared colour scale. The embedding has other axes than the
    # samples, so each panel titles its own.
    _last = sensor_trace[-1]
    mo.ui.plotly(
        plot_observations(
            [sensor_trace[0], _last, pca.transform(_last)],
            [sensor_trace.names[0], sensor_trace.names[-1], "pca"],
        )
    )
    return


@app.cell
def _(pca, sensor_trace):
    # extend() adds the embedding to the trace as one more step, linked one to one
    # by id. links_between then maps any earlier location to its place in the
    # embedding: here, sampler cell 100 and where it ended up. The embedding layer
    # gets its own axes (c0, c1) in the 3-D plot.
    mapped = pca.extend(sensor_trace)
    _to = mapped.links_between(0, -1)[:, 100].nonzero()[0]
    print(mapped.names[-1], mapped[-1].coords["c0"][_to], mapped[-1].coords["c1"][_to])
    mo.ui.plotly(mapped.plot())
    return (mapped,)


@app.cell
def _(mapped):
    # axes can be given per step: the channels (ch0 vs ch1) for the original steps,
    # and None leaves the last step at its own coordinates, c0 and c1. The blue
    # lines show where the points of the last channel layer land in the embedding.
    mo.ui.plotly(mapped.plot(axes=[(0, 3)] * (len(mapped) - 1) + [None]))
    return


@app.cell
def _(pca, sensor_trace):
    # any estimator with transform works, e.g. UMAP (needs umap-learn); only the
    # class and parameters serialize, not the fitted state

    umap = sy.Projection(UMAP(n_components=2, random_state=0)).fit(sensor_trace[0])
    print(pca.to_dict()["estimator"]["class"], umap)
    return (umap,)


@app.cell
def _(pca, sensor_trace, umap):
    # the same scatter for UMAP, next to PCA and the sample they came from
    _last = sensor_trace[-1]
    mo.ui.plotly(
        plot_observations(
            [_last, pca.transform(_last), umap.transform(_last)],
            [sensor_trace.names[-1], "pca", "umap"],
        )
    )
    return


@app.cell
def _(sensor_trace, umap):
    # extended into UMAP, with the channel layers before it
    _extended = umap.extend(sensor_trace, "umap")
    mo.ui.plotly(_extended.plot(axes=[(0, 3)] * (len(_extended) - 1) + [None]))
    return


@app.cell
def _():
    # higher dimensions means a smaller resolution and a PointSampler
    # oversized grids are refused before allocating.
    five_d = sy.Domain.from_extents([(0, 1)] * 5)
    field_5d = sy.GaussianField(five_d, length_scale=0.3, resolution=12).realize(seed=0)
    obs_5d = sy.PointSampler(1000).run(field_5d, seed=0)
    print(list(obs_5d.coords), obs_5d.values.shape)
    try:
        sy.GaussianField(five_d).realize(seed=0)
    except ValueError as err:
        print(err)
    return


@app.cell
def _(domain):
    # two clusters: the right half of the domain is offset in opposite directions on
    # the two channel groups, so PCA/UMAP see two clusters
    def _offset(sign):
        return sy.AnalyticField(lambda x, y: sign * 3.0 * (x > 5), domain)

    _fields = [
        sy.GaussianField(domain, length_scale=2.0).realize(seed=s) + _offset(sign)
        for s, sign in enumerate([1, 1, 1, -1, -1, -1])
    ]
    _corr = np.eye(6)
    _corr[:3, :3] = _corr[3:, 3:] = 0.9
    np.fill_diagonal(_corr, 1.0)
    grouped = sy.Multichannel(*_fields, corr=_corr)
    grouped_obs = sy.Observer(sy.GridSampler(48), [sy.GaussianNoise(0.3)]).run(
        grouped, seed=0
    )

    # left: channel 0 in space; right: channel 0 against channel 3 (two blobs)
    _fig = make_subplots(rows=1, cols=2, subplot_titles=["space (ch0)", "ch0 vs ch3"])
    _fig.add_trace(
        go.Heatmap(z=grouped_obs.values[..., 0], colorscale="Viridis"), row=1, col=1
    )
    _fig.add_trace(
        go.Scatter(
            x=grouped_obs.values[..., 0].ravel(),
            y=grouped_obs.values[..., 3].ravel(),
            mode="markers",
            marker={
                "size": 4,
                "color": grouped_obs.coords["x"].ravel(),
                "colorscale": "Viridis",
            },
        ),
        row=1,
        col=2,
    )
    mo.ui.plotly(_fig)
    return (grouped,)


@app.cell
def _(grouped):
    # the grouped data through an observer: every step is kept in the trace
    grouped_trace = sy.Observer(
        sy.GridSampler(48),
        [sy.GaussianNoise(0.3), sy.RandomDropout(0.1), sy.Quantize(levels=6)],
    ).trace(grouped, seed=0)
    grouped_trace.names
    return (grouped_trace,)


@app.cell
def _(grouped_trace):
    # fit PCA on the sampled layer, then add the embedding as the last step

    grouped_pca = sy.Projection(PCA(n_components=2)).fit(grouped_trace[0])
    grouped_embedded = grouped_pca.extend(grouped_trace)
    grouped_embedded.names
    return (grouped_embedded,)


@app.cell
def _(grouped_embedded):
    # every layer: channels 0 vs 3 for the observer steps, the embedding at the top
    mo.ui.plotly(
        grouped_embedded.plot(axes=[(0, 1)] * (len(grouped_embedded) - 1) + [None])
    )
    return


@app.cell
def _(grouped_embedded):
    # plotting the trace in its coordinate space
    mo.ui.plotly(grouped_embedded.plot())
    return


if __name__ == "__main__":
    app.run()
