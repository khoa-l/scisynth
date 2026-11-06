import marimo

__generated_with = "0.25.2"
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

    from scisynth.latent.primatives import ellipse_mask, rectangle_mask, apply_mask, flatten_masks


@app.cell
def _():
    domain = sy.Domain.from_extents([(0, 100), (0, 50)], units=["mm", "mm"])
    generator = sy.GaussianField(domain, length_scale=1.5, amplitude=stats.uniform(0.5, 1.0))
    simpleLatent = generator.realize(seed=17)
    newLatent = generator.realize(seed=27)

    fields = [generator.realize(seed=s) for s in [7,37,67]]
    correlation = [[1.0,.8,.0],[.8,1.0,.0],[.0,.0,1.0]]
    convolutedLatent = sy.Multichannel(*fields, corr=correlation)

    ellipseLatent = ellipse_mask(domain, (200,100), (50,25), (30,15))
    rectangleLatent = rectangle_mask(domain, (200,100), (50,25), (30,15))

    complicatedLatent = apply_mask(ellipseLatent, simpleLatent, newLatent)
    return (
        complicatedLatent,
        convolutedLatent,
        domain,
        ellipseLatent,
        generator,
        rectangleLatent,
        simpleLatent,
    )


@app.cell
def _(
    complicatedLatent,
    convolutedLatent,
    ellipseLatent,
    rectangleLatent,
    simpleLatent,
):
    xBound,yBound = 150,60

    grid = sy.GridSampler(shape=(xBound,yBound))
    pnts = sy.PointSampler(n=xBound*yBound)

    mo.vstack([
        mo.ui.plotly(grid.plot(simpleLatent)),
        mo.ui.plotly(pnts.plot(simpleLatent)),
        mo.ui.plotly(grid.plot(convolutedLatent)),
        mo.ui.plotly(pnts.plot(convolutedLatent)),
        mo.ui.plotly(grid.plot(ellipseLatent)),
        mo.ui.plotly(pnts.plot(ellipseLatent)),
        mo.ui.plotly(grid.plot(rectangleLatent)),
        mo.ui.plotly(pnts.plot(rectangleLatent)),
        mo.ui.plotly(grid.plot(complicatedLatent)),
        mo.ui.plotly(pnts.plot(complicatedLatent))
    ])
    return grid, pnts


@app.cell
def _(domain, generator, grid, pnts):
    # How convoluted?
    noises = [generator.realize(seed=s) for s in [1,2,3,5,8,13,21,34,55]]

    mickeyLeftEar = ellipse_mask(domain, (100,100),(25,35), (15,15))
    mickeyRightEar = ellipse_mask(domain, (100,100), (75,35), (15,15))
    mickeyFace = ellipse_mask(domain, (200,200), (50,25), (30,20))
    mickeyLeftEye = ellipse_mask(domain, (50,50), (35,30), (7,7))
    mickeyRightEye = ellipse_mask(domain, (50,50), (65, 30), (7,7))
    mickeySmile = rectangle_mask(domain, (100,50), (50,15), (25, 5))

    mickeyHead = flatten_masks([mickeyLeftEar, mickeyRightEar, mickeyFace])
    mickeyFace = flatten_masks([mickeyLeftEye, mickeyRightEye, mickeySmile])
    mickeyMask = flatten_masks([mickeyHead, mickeyFace], add=False)

    mickeyLatent = apply_mask(mickeyMask, noises[0], noises[1])

    mo.vstack([
        mo.ui.plotly(grid.plot(mickeyMask)),
        mo.ui.plotly(grid.plot(mickeyLatent)),
        mo.ui.plotly(pnts.plot(mickeyLatent))
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
