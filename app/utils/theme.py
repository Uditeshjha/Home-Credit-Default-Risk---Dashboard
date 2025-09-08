"""
Corporate 'warm' theme helper.

Call `apply_theme()` at the top of every page.
This sets seaborn/matplotlib defaults and exposes a small palette dict to use in plots.
"""

# utils/theme.py (update)
import seaborn as sns
import matplotlib as mpl

# Core corporate colors (kept stable)
PALETTE = {
    "navy": "#0B2545",
    "warm_orange": "#FF7A1A",
    "gold": "#F4C95D",
    "beige": "#FAF4E6",     # light wrapper, but charts use white interior
    "muted_gray": "#6E6E6E",
    "text": "#0B2545"       # use navy text for strong contrast on white chart area
}

def get_palette():
    """Return the static palette. Kept simple so pages can call PALETTE = get_palette()."""
    return PALETTE

def apply_theme():
    """
    Apply seaborn/matplotlib defaults.
    IMPORTANT: This sets charts to use a white / light interior so they remain readable
    in both Streamlit light and dark app chrome. This is intentional and permanent.
    """
    pal = get_palette()

    # Seaborn defaults
    sns.set_theme(style="whitegrid")
    sns.set_palette([pal["navy"], pal["beige"], pal["gold"], pal["muted_gray"]])

    # Matplotlib rcParams - neutral, high-contrast settings
    mpl.rcParams.update({
        # Make the figure wrapper slightly warm (optional); axes interior white for consistency
        "figure.facecolor": pal["beige"],    # surrounding area (blends with light-themed pages)
        "axes.facecolor": "white",           # KEEP chart interior white for clarity
        "savefig.facecolor": pal["beige"],

        # Text / tick colors use navy for high contrast on white axes
        "axes.labelcolor": pal["text"],
        "xtick.color": pal["text"],
        "ytick.color": pal["text"],
        "text.color": pal["text"],

        "axes.edgecolor": pal["muted_gray"],
        "legend.frameon": False,
        "grid.color": "#E8E8E8",
        "font.size": 11
    })