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
    longGen = sy.GaussianField(domain, length_scale=5, amplitude=stats.uniform(0.25,1.0))
    simpleLatent = generator.realize(seed=17)
    newLatent = generator.realize(seed=27)

    noises = [generator.realize(seed=s) for s in [1,2,3,5,8,13,21,34,55]]
    sounds = [longGen.realize(seed=s) for s in [1,2,3,5,8,13,21,34,55]]

    fields = noises[0:3]
    correlation = [[1.0,.8,.0],[.8,1.0,.0],[.0,.0,1.0]]
    convolutedLatent = sy.Multichannel(*fields, corr=correlation)

    rectangleLatent = rectangle_mask(domain, (200,100), (50,25), (30,15))
    ellipseLatent = ellipse_mask(domain, (200,100), (50,25), (30,15), order=1)

    complicatedLatent = apply_mask(ellipseLatent, simpleLatent, newLatent)
    return (
        complicatedLatent,
        convolutedLatent,
        domain,
        ellipseLatent,
        noises,
        rectangleLatent,
        simpleLatent,
        sounds,
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
        mo.ui.plotly(grid.plot(rectangleLatent)),
        mo.ui.plotly(pnts.plot(rectangleLatent)),
        mo.ui.plotly(grid.plot(ellipseLatent)),
        mo.ui.plotly(pnts.plot(ellipseLatent)),
        mo.ui.plotly(grid.plot(complicatedLatent)),
        mo.ui.plotly(pnts.plot(complicatedLatent))
    ])
    return grid, pnts


@app.cell
def _(domain, grid, noises, pnts, sounds):
    # How convoluted?
    mickeyLeftEar = ellipse_mask(domain, (100,100),(25,35), (15,15))
    mickeyRightEar = ellipse_mask(domain, (100,100), (75,35), (15,15))
    mickeyFace = ellipse_mask(domain, (200,200), (50,25), (30,20))
    mickeyLeftEye = ellipse_mask(domain, (50,50), (35,30), (7,7),order=2)
    mickeyRightEye = ellipse_mask(domain, (50,50), (65, 30), (7,7), order=2)
    mickeySmile = rectangle_mask(domain, (100,50), (50,15), (25, 5), order=5)

    mickeyHead = flatten_masks([mickeyLeftEar, mickeyRightEar, mickeyFace])
    mickeyFace = flatten_masks([mickeyLeftEye, mickeyRightEye, mickeySmile])
    mickeyMask = flatten_masks([mickeyHead, mickeyFace], add=False)

    mickeyLatent = apply_mask(mickeyMask, noises[0], sounds[1])

    mo.vstack([
        mo.ui.plotly(grid.plot(mickeyMask)),
        mo.ui.plotly(grid.plot(mickeyLatent)),
        mo.ui.plotly(pnts.plot(mickeyLatent))
    ])
    return


@app.cell
def _(domain, grid, noises, pnts, sounds):
    ufoBase = ellipse_mask(domain, (200,200),(50,20), (30,8))
    ufoTop = ellipse_mask(domain, (200,200), (50,30), (20,15))
    ufoDots = [ellipse_mask(domain, (50,50), (x,y), (1.5,1.5))
    for (x,y) in [(25,20),(30,18),(35,16),(40,15),(45,15),(50,15),(55,15),(60,15),(65,16),(70,18),(75,20)]]
    ufoWindshield = ellipse_mask(domain, (200,200),(50,30),(17,12))
    ufoWindframe = ellipse_mask(domain, (100,100),(50,28),(17,5))

    ufoBackground = flatten_masks([ufoBase, ufoTop])
    ufoDotMask = flatten_masks([ufoDot for ufoDot in ufoDots])
    ufoWindow = flatten_masks([flatten_masks([ufoWindshield,ufoBase], add=False),ufoWindframe])
    ufoIsh = flatten_masks([ufoBackground, ufoDotMask], add=False)
    ufoMask = flatten_masks([ufoIsh,ufoWindow], add=False)

    ufoLatent = apply_mask(ufoMask, noises[3], sounds[2])

    mo.vstack([
        mo.ui.plotly(grid.plot(ufoMask)),
        mo.ui.plotly(grid.plot(ufoLatent)),
        mo.ui.plotly(pnts.plot(ufoLatent))
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
