"""
common/viz.py -- the shared figure palette and matplotlib style.

One palette for every figure in the report, so the three members' networks
look like one document rather than three. Imported by all three members
(development_guidelines.md §3); never copy-pasted into a member's src/.

    from common import viz
    viz.apply_style()               # once, at the top of the notebook
    viz.BLOCK_COLORS["T"]           # never a literal hex in a notebook

Colours are the validated categorical slots 1-4 (blue / orange / aqua /
yellow) plus a blue<->red diverging pair with a grey midpoint. Slot order is
fixed and never cycled -- assign colours explicitly, and never repurpose a
block colour for something else.

BLOCK_NAMES lives here and nowhere else, so that exported table labels and
figure legends cannot drift apart.
"""

import matplotlib as mpl

# Categorical slots, fixed order (validated: worst adjacent CVD dE 9.1).
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# Survey blocks always get the same colour across all members' figures.
BLOCK_COLORS = {"T": CAT[0], "E": CAT[1], "S": CAT[2], "V": CAT[3]}
BLOCK_NAMES = {"T": "Technology", "E": "Education", "S": "Society/Ethics", "V": "Environment"}

# Diverging pair for signed quantities (gap, correlation, deviation).
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"

INK, INK_SOFT = "#0b0b0b", "#52514e"

# Recessive structure: axis lines, grid, error bars, anything that is not
# data. Kept desaturated so the data is the only saturated thing on the page.
AXIS, AXIS_SOFT = "#d6d5d1", "#ececea"

# Null / reference distributions. Every member compares something against a
# null (ER, configuration model, permutation, sign shuffle), and the null
# always gets this grey so it reads as the baseline rather than a series.
NULL = "#b8b7b3"


def apply_style() -> None:
    """Recessive axes, light grid, text in ink tokens rather than series colour."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": AXIS_SOFT,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "text.color": INK,
        "font.size": 9,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
