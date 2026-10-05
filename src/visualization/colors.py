"""
src/visualization/colors.py
============================

Centralised color palette for the counterfactual CGM-SLH project.

All figure modules import their colour constants from here so that the
visual identity can be updated in one place.  The palette is built on
the Wong (2011) color-blind-safe set and extended with light/dark
variants for fills and emphasis.

Sections
--------
Timestep colors       C_5MIN, C_15MIN
Coverage colors       COLOR_PRIMARY, COLOR_THRESHOLD, COLOR_ANNOTATION, COLOR_BOX_BG
Metrics colors        CV_GRADIENT, TIR_GRADIENT, DIVERGING, MEAN_GRADIENT, CATEGORICAL_5
Filtering colors      COLOR_INCLUDED, COLOR_EXCLUDED, COLOR_ZONE, COLORMAP_CONTINUOUS
Episode colors        COLOR_SLH, COLOR_LLH, COLOR_EXCLUDED
AGP colors            COLOR_AGP_*, COLOR_THRESHOLD_HYPER, COLOR_THRESHOLD_HYPO
SLH schema colors     COLOR_BEFORE_WINDOW, COLOR_AFTER_WINDOW, FUNNEL_COLORS, STAGE_COLORS
Scatter/counterfactual colors  COLOR_BENEFICIAL, COLOR_DETRIMENTAL, OFFSET_COLORS
"""

from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------------
# Timesteps colors
# ----------------------------------------------------------------------------
C_5MIN   = "#003865"   # navy
C_15MIN  = "#C9A84C"   # warm gold

# ----------------------------------------------------------------------------
# Coverage colors
# ----------------------------------------------------------------------------
COLOR_PRIMARY     = "#003865"   # Navy blue — primary data
COLOR_THRESHOLD   = "#B5121B"   # Red — threshold line
COLOR_ANNOTATION  = "#6D6E71"   # Grey — annotations and labels
COLOR_BOX_BG      = "white"     # Background for text boxes

# ----------------------------------------------------------------------------
# Metrics colors
# ----------------------------------------------------------------------------
# 4. CV gradient (low → high variability)
CV_GRADIENT = LinearSegmentedColormap.from_list(
    'cv',
    ['#F7FBFF', '#FDAE61', '#D73027']  # light → orange → red
)

# 3. TIR gradient (bad → neutral → good)
TIR_GRADIENT = LinearSegmentedColormap.from_list(
    'tir',
    ['#B2182B', '#FDDDBC', '#1B7837']  # red → light → green
)

DIVERGING = LinearSegmentedColormap.from_list(
    'diverging',
    ['#2166AC', '#F7F7F7', '#B2182B']  # blue → white → red
)

MEAN_GRADIENT = LinearSegmentedColormap.from_list(
    'mean',
    ['#1B7837', '#FEE08B', '#D73027']  # green → yellow → red
)

WONG_COLORS = {
    'black':          '#000000',
    'orange':         '#E69F00',
    'sky_blue':       '#56B4E9',
    'bluish_green':   '#009E73',
    'yellow':         '#F0E442',
    'blue':           '#0072B2',
    'vermillion':     '#D55E00',
    'reddish_purple': '#CC79A7',
}

CATEGORICAL_5 = [
    WONG_COLORS['blue'],
    WONG_COLORS['orange'],
    WONG_COLORS['bluish_green'],
    WONG_COLORS['vermillion'],
    WONG_COLORS['reddish_purple'],
]
# ----------------------------------------------------------------------------
# Filtering colors
# ----------------------------------------------------------------------------
COLOR_INCLUDED = "#0072B2"   # Blue — selected patients
COLOR_EXCLUDED = "#cccccc"   # Grey — excluded patients
# COLOR_THRESHOLD = "#D55E00"  # Red — threshold boundary lines
COLOR_ZONE = "#56B4E9"       # Blue — inclusion zone shading

COLORMAP_CONTINUOUS = "cividis"

# ----------------------------------------------------------------------------
# Episode colors
# ----------------------------------------------------------------------------

# Classification colors
COLOR_SLH = "#1F4E79"          # Dark navy — confirmed SLH
COLOR_LLH = "#E8A0A8"          # Soft rose — LLH (duration > D_MAX_MIN)
COLOR_EXCLUDED = "#AAAAAA"     # Mid-grey — excluded episodes

# ----------------------------------------------------------------------------
# AGP colors
# ----------------------------------------------------------------------------

# AGP colors
COLOR_AGP_PERCENTILE_95 = "#56B4E9"   # Sky blue — 5-95 percentile band
COLOR_AGP_PERCENTILE_IQR = "#0072B2"  # Blue — IQR band
COLOR_AGP_MEDIAN = "#1A56A0"          # Dark blue — median line
COLOR_AGP_MEAN = "#009E73"            # Green — mean line
COLOR_AGP_TARGET_ZONE = "#009E73"     # Green — target zone shading

# Threshold colors
COLOR_THRESHOLD_HYPER = "#D55E00"     # Red-orange — hyperglycemia line
COLOR_THRESHOLD_HYPO = "#E69F00"      # Orange — hypoglycemia line

# Duration bin colors (light → dark gradient)
COLOR_DURATION_BINS = [
    "#FFE5CC",   # 0–15 min (lightest)
    "#FFB380",   # 15–30 min
    "#FF9966",   # 30–45 min
    "#FF8066",   # 45–60 min
    "#E8604C",   # 60–75 min
    "#CC3333",   # 75–90 min
    "#8B0000",   # 90+ min (darkest)
]

COLOR_RECURRENCE_THRESHOLD = "#1F4E79"  # Navy — systematic threshold line


# Table colors
COLOR_TABLE_HEADER = "#4A90E2"        # Blue — table header
COLOR_TABLE_HIGHLIGHT = "#FFE5CC"     # Light orange — highlighted row
COLOR_TABLE_ALT_ROW = "#F0F0F0"       # Light grey — alternate rows


# ----------------------------------------------------------------------------
# SLH schema colors
# ----------------------------------------------------------------------------
COLOR_BEFORE_WINDOW = "#E69F00"   # Gold — before-window shading
COLOR_AFTER_WINDOW = "#009E73"    # Teal — after-window shading


# Two distinct night roles — do not substitute one for the other:
#   COLOR_NIGHT          background shading band for the night window (always
#                        drawn under the data, usually with alpha)
#   COLOR_NIGHT_CATEGORY foreground colour for the "Night" EPISODE CATEGORY
#                        (bars, markers, annotations). It must stay dark: in
#                        the onset panel of Figure 2 these markers are plotted
#                        on top of the COLOR_NIGHT band, so a pale category
#                        colour would be invisible there.
COLOR_NIGHT = "#DBD0D0"          # Pale blue-grey — night shading
COLOR_NIGHT_CATEGORY = "#2c3e50"  # Dark navy — "Night" episode category
COLOR_TRACE = "#999999"        # Light-grey — raw CGM trace
COLOR_HYPER = "#E69F00"        # Orange — H_HYPER line
COLOR_CONTEXT = "#009E73"      # Teal — H_CONTEXT line
COLOR_HYPO = "#BBBBBB"         # Light grey — H_HYPO line


# Color palette for funnel stages
FUNNEL_COLORS = {
    "C1": "#888888",      # Dark grey - duration bounds
    "C2": "#D3D3D3",      # Light grey - data completeness
    "C3": "#BBBBBB",      # Medium grey - daytime restriction
    "C4": "#E69F00",      # Orange - no pre-hyperglycemia
    "C5": "#F0E442",      # Yellow - no post-hyperglycemia
    "SLH": "#0072B2",     # Navy blue - passed all criteria
}

# Colors for SLH crossing breakdown
CROSSING_COLORS = {
    "before_only": "#E69F00",   # Orange
    "after_only": "#009E73",    # Teal
    "both": "#0072B2",          # Navy blue
}

# Color palette - one color per classification level
STAGE_COLORS = {
    "C1_duration":     "#D55E00",  # Vermillion
    "C2_incomplete":   "#CC79A7",  # Reddish purple
    "C3_nocturnal":    "#999999",  # Neutral gray (outside Wong core, useful special case)
    "C4_hyper_before": "#E69F00",  # Orange
    "C5_hyper_after":  "#F0E442",  # Yellow
    "C6_no_cross":     "#56B4E9",  # Sky blue
    "SLH":             "#009E73",  # Bluish green (success/pass)
}
# ----------------------------------------------------------------------------
# Quartile colors (light → dark blue gradient, Q1 lowest → Q4 highest n_SLH)
# ----------------------------------------------------------------------------
QUARTILE_COLORS = {
    "Q1": "#d4e6f1",   # Very light blue
    "Q2": "#a9cce3",   # Light blue
    "Q3": "#7fb3d5",   # Medium blue
    "Q4": "#5499c7",   # Darker blue
}

# ----------------------------------------------------------------------------
# Scatter analysis colors
# ----------------------------------------------------------------------------
COLOR_BENEFICIAL = "#009E73"   # Green — beneficial metric change
COLOR_DETRIMENTAL = "#D55E00"  # Red/orange — detrimental metric change
COLOR_NEUTRAL = "#999999"      # Grey — neutral/minimal change
COLOR_REGRESSION = "#0072B2"   # Blue — OLS regression line

# ----------------------------------------------------------------------------
# Counterfactual colors
# ----------------------------------------------------------------------------
# Counterfactual
OFFSET_COLORS = {
    0:  "#999999",  # Light grey
    15: "#56B4E9",  # Sky blue
    30: "#0072B2",  # Blue
    45: "#CC79A7",  # Reddish purple
    60: "#E69F00",  # Orange
    75: "#009E73",  # Bluish green
    90: "#D55E00",  # Vermilion
}

COLOR_ORIGINAL = "#333333"        # Dark grey
COLOR_COUNTERFACTUAL = "#0072B2"  # Blue — C(t) final counterfactual
COLOR_SLH_HIGHLIGHT = "#E69F00"   # Orange — SLH episode highlighting
COLOR_ZERO_LINE = "#888888"       # Grey — zero-reference line in sensitivity plots
COLOR_LININTERP = "#E69F00"       # Orange — L(t) linear interpolation baseline
COLOR_ANCHOR = "#D55E00"          # Vermilion — local-minima anchor points
COLOR_EPISODE = "#E69F00"         # Orange — episode core shading

# =============================================================================
# EXTENDED WONG PALETTE (tints + shades + neutrals)
# =============================================================================
# Built from original Wong colors to increase palette size while preserving
# visual coherence and colorblind safety.

# --- Base Wong colors ---
# black             #000000
# orange            #E69F00
# sky_blue          #56B4E9
# bluish_green      #009E73
# yellow            #F0E442
# blue              #0072B2
# vermillion        #D55E00
# reddish_purple    #CC79A7

# --- Light variants (for fills, backgrounds) ---
# light_orange        #F8C57A
# light_sky_blue      #A6D4F2
# light_bluish_green  #6FD1B5
# light_yellow        #FFF3A6
# light_blue          #6AAED6
# light_vermillion    #F29A7A
# light_purple        #E3A6C9

# --- Dark variants (for lines, emphasis) ---
# dark_orange         #B37500
# dark_sky_blue       #2F80B9
# dark_bluish_green   #006D5B
# dark_yellow         #C9B800
# dark_blue           #004C7A
# dark_vermillion     #9A3E00
# dark_purple         #8E4F73

# --- Neutral extensions (important for medical plots) ---
# dark_grey           #4D4D4D
# mid_grey            #7F7F7F
# light_grey          #BFBFBF
# very_light_grey     #E6E6E6

# --- Cool/warm balance additions (Lancet-style compatible) ---
# deep_teal           #005F5F
# soft_teal           #4DB6AC
# muted_blue          #4C78A8
# muted_red           #B04A4A
# soft_gold           #D4AF37
# pale_sand           #F5E6C5

# Notes:
# - Use LIGHT variants for areas (IQR, CI bands, fills)
# - Use BASE colors for main categories
# - Use DARK variants for lines / medians / emphasis
# - Avoid mixing too many saturated colors in one figure
# =============================================================================