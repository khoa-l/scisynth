"""Plotting with plotly, registered by subject and kind.

Optional: ``pip install "scisynth[viz]"`` (plotly). Importing this
package does not import plotly: each plot function imports it when called.
"""

from .plot import plot_latent, plot_observations
from .registry import (
    NoPlotError,
    get_plot,
    kinds_of,
    plot,
    register_plot,
    registered_plots,
    resolve_plot,
    subjects_of,
)

__all__ = [
    "NoPlotError",
    "get_plot",
    "kinds_of",
    "plot",
    "plot_latent",
    "plot_observations",
    "register_plot",
    "registered_plots",
    "resolve_plot",
    "subjects_of",
]
