import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Positional Fluidity Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# Industry-style refined minimalism. Single neutral background, narrow
# accent palette, generous whitespace, consistent typography. Inspired by
# professional analytics platforms (Bloomberg / StatsBomb / Opta aesthetic).
# ══════════════════════════════════════════════════════════════════════════════

# Core palette
PALETTE = {
    'bg':        '#FAFAF7',   # warm off-white background
    'panel':     '#FFFFFF',   # card / panel
    'border':    '#E5E5E0',   # subtle border
    'ink':       '#1A1A1A',   # primary text
    'ink_soft':  '#5C5C5C',   # secondary text
    'ink_muted': '#8A8A8A',   # tertiary text
    'accent':    '#1F4E5F',   # deep teal — primary data colour
    'fluid':     '#2E7D5B',   # muted green — fluid / above zero
    'positional':'#B33A3A',   # muted red — positional / below zero
    'rule':      '#D4D4CE',   # divider lines
    'grid':      '#EAEAE3',   # plot gridlines
}

# Matplotlib defaults — keep all plots visually consistent
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.edgecolor': PALETTE['border'],
    'axes.linewidth': 0.8,
    'axes.facecolor': PALETTE['bg'],
    'figure.facecolor': PALETTE['bg'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': PALETTE['ink_soft'],
    'ytick.color': PALETTE['ink_soft'],
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.frameon': False,
    'legend.fontsize': 10,
})

# Streamlit CSS — applied globally
st.markdown(f"""
<style>
    .stApp {{
        background-color: {PALETTE['bg']};
    }}
    [data-testid="stSidebar"] {{
        background-color: {PALETTE['panel']};
        border-right: 1px solid {PALETTE['border']};
    }}
    [data-testid="stSidebar"] * {{
        color: {PALETTE['ink']} !important;
    }}
    html, body, p, li, span, label, div {{
        color: {PALETTE['ink']};
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    h1 {{
        color: {PALETTE['ink']} !important;
        font-weight: 700;
        letter-spacing: -0.02em;
        font-size: 2.1rem !important;
    }}
    h2 {{
        color: {PALETTE['ink']} !important;
        font-weight: 600;
        letter-spacing: -0.01em;
        font-size: 1.4rem !important;
        margin-top: 0.6rem;
    }}
    h3, h4 {{
        color: {PALETTE['ink']} !important;
        font-weight: 600;
    }}
    hr {{
        border: none;
        border-top: 1px solid {PALETTE['rule']};
        margin: 1.4rem 0;
    }}
    .section-header {{
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {PALETTE['ink_soft']};
        border-bottom: 1px solid {PALETTE['rule']};
        padding-bottom: 6px;
        margin: 1.2rem 0 0.9rem 0;
    }}
    .info-card {{
        background-color: {PALETTE['panel']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        padding: 18px 22px;
        margin-bottom: 12px;
        line-height: 1.55;
    }}
    .info-card h4 {{
        font-size: 0.95rem;
        font-weight: 700;
        color: {PALETTE['ink']};
        margin-top: 14px;
        margin-bottom: 4px;
    }}
    .info-card h4:first-of-type {{
        margin-top: 0;
    }}
    .info-card p {{
        font-size: 0.92rem;
        color: {PALETTE['ink_soft']};
        margin-bottom: 6px;
    }}
    .stat-box {{
        background-color: {PALETTE['panel']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        padding: 14px 16px;
    }}
    .stat-label {{
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {PALETTE['ink_muted']};
        margin-bottom: 6px;
    }}
    .stat-value {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {PALETTE['ink']};
        line-height: 1.1;
    }}
    .stSelectbox label {{
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {PALETTE['ink_soft']} !important;
    }}
    div[data-baseweb="select"] {{
        background-color: {PALETTE['panel']} !important;
        border: 1px solid {PALETTE['border']} !important;
        border-radius: 4px !important;
    }}
    div[data-baseweb="select"] * {{
        color: {PALETTE['ink']} !important;
    }}
    button[data-baseweb="tab"] {{
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: {PALETTE['ink_soft']} !important;
        padding: 10px 22px !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {PALETTE['accent']} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {PALETTE['ink']} !important;
    }}
    code {{
        background-color: {PALETTE['bg']};
        border: 1px solid {PALETTE['border']};
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 0.85em;
    }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Google Drive ID for the large touches parquet file
TOUCHES_GDRIVE_ID = "1EGNoqoQJXe8aOjFe5BUw2ya41Yj6eBYg"


@st.cache_data
def load_data():
    """Load the four small parquet files bundled with the app."""
    player_season = pd.read_parquet(os.path.join(SCRIPT_DIR, "p.season.parquet"))
    team_match    = pd.read_parquet(os.path.join(SCRIPT_DIR, "t.match.parquet"))
    team_season   = pd.read_parquet(os.path.join(SCRIPT_DIR, "t.season.parquet"))
    player_match  = pd.read_parquet(os.path.join(SCRIPT_DIR, "p.match.parquet"))
    return player_season, team_match, team_season, player_match


@st.cache_data(show_spinner="Loading touch data...")
def load_touches(player_id):
    """Fetch the large touches parquet from Google Drive on first run, cache locally."""
    import gdown
    output_path = "/tmp/pf_touches.parquet"
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={TOUCHES_GDRIVE_ID}"
        gdown.download(url, output_path, quiet=False)
    touches = pd.read_parquet(output_path)
    return touches[touches['player_id'] == player_id]


try:
    player_season_stats, team_match_stats, team_season_stats, player_match_stats = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Data loading failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════
X_BINS = [0, 40, 80, 120]
Y_BINS = [0, 26.7, 53.3, 80]

POSITION_GROUPS = {
    'Defence':  ['Center Back', 'Full-Back'],
    'Midfield': ['Defensive Midfield', 'Central Midfield', 'Attacking Midfield', 'Wide Midfield'],
    'Attack':   ['Center Forward', 'Wide Forward'],
}

ZONAL_COLS = {
    'Defence':  'defence_avg_z',
    'Midfield': 'midfield_avg_z',
    'Attack':   'attack_avg_z',
}

ZONAL_MATCH_COLS = {
    'Defence':  'defence_z',
    'Midfield': 'midfield_z',
    'Attack':   'attack_z',
}


def assign_zone_9(x, y):
    x_zone = np.digitize(np.array(x), X_BINS) - 1
    y_zone = np.digitize(np.array(y), Y_BINS) - 1
    x_zone = np.clip(x_zone, 0, len(X_BINS) - 2)
    y_zone = np.clip(y_zone, 0, len(Y_BINS) - 2)
    return x_zone * 3 + y_zone + 1


def stat_box(label, value):
    """Render a flat stat box with an uppercase label and bold value."""
    return (
        f"<div class='stat-box'>"
        f"<div class='stat-label'>{label}</div>"
        f"<div class='stat-value'>{value}</div>"
        f"</div>"
    )


def draw_binned_touch_map(touch_df, ax):
    pitch = Pitch(pitch_type='statsbomb',
                  line_color=PALETTE['ink_soft'],
                  pitch_color=PALETTE['panel'],
                  linewidth=1)
    pitch.draw(ax=ax)

    if len(touch_df) == 0:
        ax.text(60, 40, 'No touch data', ha='center', va='center',
                fontsize=11, color=PALETTE['ink_muted'])
        return

    touch_df = touch_df.copy()
    touch_df['zone_9'] = assign_zone_9(touch_df['x'].values, touch_df['y'].values)
    zone_pcts = touch_df['zone_9'].value_counts(normalize=True) * 100

    # Use a monochrome teal ramp so heatmap is consistent with overall palette
    teal_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        'teal_heat',
        ['#F2F0EB', '#A7C2C8', '#5E8B95', PALETTE['accent']]
    )

    for zone_num in range(1, 10):
        col = (zone_num - 1) // 3
        row = (zone_num - 1) % 3
        x0, x1 = X_BINS[col], X_BINS[col + 1]
        y0, y1 = Y_BINS[row], Y_BINS[row + 1]
        pct = zone_pcts.get(zone_num, 0)
        color = teal_cmap(min(pct / 40, 1.0))
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                             color=color, zorder=2, alpha=0.9)
        ax.add_patch(rect)

        text_color = 'white' if pct >= 18 else PALETTE['ink']
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f'{pct:.0f}%',
                ha='center', va='center', fontsize=11,
                fontweight='bold', color=text_color, zorder=4)

    for x in X_BINS[1:-1]:
        ax.plot([x, x], [0, 80], color=PALETTE['ink'],
                linewidth=1, zorder=3, linestyle='--', alpha=0.7)
    for y in Y_BINS[1:-1]:
        ax.plot([0, 120], [y, y], color=PALETTE['ink'],
                linewidth=1, zorder=3, linestyle='--', alpha=0.7)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown(
    f"<div style='font-size:0.75rem; font-weight:700; letter-spacing:0.08em; "
    f"text-transform:uppercase; color:{PALETTE['ink_muted']}; margin-bottom:10px;'>"
    f"Navigation</div>",
    unsafe_allow_html=True
)
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Player Statistics", "Team Statistics"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style='font-size:0.78rem; color:{PALETTE['ink_soft']}; line-height:1.6;'>
<div style='font-weight:700; color:{PALETTE['ink']}; margin-bottom:6px;'>Data Source</div>
StatsBomb Open Data<br><br>
<div style='font-weight:700; color:{PALETTE['ink']}; margin-bottom:6px;'>Season</div>
2015–16<br><br>
<div style='font-weight:700; color:{PALETTE['ink']}; margin-bottom:6px;'>Leagues</div>
Premier League<br>La Liga<br>Bundesliga<br>Serie A
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":

    st.markdown("# Positional Fluidity Dashboard")
    st.markdown(
        f"<div style='color:{PALETTE['ink_soft']}; font-size:1.05rem; "
        f"margin-top:-6px;'>Team and player statistics — Premier League, La Liga, "
        f"Bundesliga, Serie A · 2015–16</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("## What is Positional Fluidity?")
        st.markdown("""
        <div class='info-card'>
        <p>Throughout its history, debates surrounding fluidity and structure have shaped tactical thinking in football.
        This has been characterised by the contrast between tactical systems championing <b>positional fluidity</b>,
        characterised by players having creative freedom to move around the pitch, and more organised, structured football.</p>
        <p><b>Positional fluidity</b>, as measured in this project, refers to the degree to which a player operates
        across multiple areas of the pitch, instead of occupying rigid zones of attacking activity. This is expressed
        as the percentage of a player's touches outside the most common zone of the pitch in which they touch the ball.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("## Why Does It Matter?")
        st.markdown("""
        <div class='info-card'>
        <p>Positional fluidity currently sits at the heart of the tactical debate in modern football. For most of the
        twenty-first century, <b>positional play</b> (<i>juego de posición</i>), an attacking approach centred upon
        players operating in rigid pre-defined zones, has been globally dominant, championed by high-profile managers
        such as Pep Guardiola and Mikel Arteta. In recent years, positional play has been challenged by more
        positionally fluid and expressive approaches, often grouped under the term <b>relational play</b>.</p>
        <p>Despite its modern significance, data-driven research of positional fluidity is limited, with the bulk of
        information coming from qualitative observations. This dashboard displays a data-driven exploration of
        positional fluidity at a player and team level, to try and improve empirical understandings of fluidity as
        a footballing concept.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## How Were the Fluidity Scores Calculated?")

    tab1, tab2 = st.tabs(["Player Level", "Team Level"])

    # ── PLAYER-LEVEL EXPLANATION ────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class='info-card'>

        <h4>1 — Extracting Open-Play Touches</h4>
        <p>Outfield, open-play ball touch data was extracted for every match in the dataset. Goalkeeper touches and
        set-piece events (corners, free-kicks, penalties, throw-ins) were removed, ensuring all data captured
        attacking movement during live play. For inclusion in match-level scoring, a player was required to record
        a minimum of <b>fifteen open-play touches in a single position within a single match</b>. This threshold
        filters out noise from substitutions, brief positional changes, red cards and injuries, while preserving
        enough volume to capture a meaningful movement profile.</p>

        <h4>2 — Zone Assignment</h4>
        <p>The pitch was divided into a <b>nine-zone grid</b> (see touch maps in player statistics for reference),
        splitting the defensive, midfield and attacking thirds into left, central and right channels. StatsBomb
        coordinates are normalised to a consistent left-to-right attacking direction throughout each match, which
        is preserved here.</p>

        <h4>3 — Raw Fluidity Percentage</h4>
        <p>For each qualifying player-position-match combination, the <b>fluidity score</b> was calculated as the
        percentage of touches falling outside the player's most frequently occupied zone:
        <code>(touches outside modal zone / total touches) × 100</code>. Higher scores indicate more fluid,
        zone-crossing player movements.</p>

        <h4>4 — Position Z-Scoring</h4>
        <p>Match-level fluidity percentages were converted into <b>z-scores</b> within each position category. A
        z-score expresses how far a value sits from the average of its reference group, measured in standard
        deviations: a z-score of zero is exactly average, +1.0 sits one standard deviation above the mean, and
        -1.0 one below. Standardising within position controls for the natural variance in movement profile
        between positions (a centre back's baseline fluidity is far lower than a wide forward's), allowing direct
        and meaningful comparison across positions on a common scale.</p>

        <h4>5 — Seasonal Aggregation</h4>
        <p>To produce season-level player statistics, a player was required to meet the fifteen-touch threshold in
        at least <b>eight matches within a single position</b>. Seasonal fluidity scores are the mean of all
        qualifying match-level z-scores. A player may appear in the dashboard multiple times if they qualified
        in more than one distinct position across the season.</p>

        </div>
        """, unsafe_allow_html=True)

        # Reference pitch grid + positions table
        col_grid, col_pos = st.columns([1, 1])

        with col_grid:
            pitch_home = Pitch(pitch_type='statsbomb',
                               line_color=PALETTE['ink_soft'],
                               pitch_color=PALETTE['panel'],
                               linewidth=1)
            fig_grid, ax_grid = plt.subplots(figsize=(7, 4.5))
            fig_grid.patch.set_facecolor(PALETTE['bg'])
            ax_grid.set_facecolor(PALETTE['bg'])
            pitch_home.draw(ax=ax_grid)
            for x in X_BINS[1:-1]:
                ax_grid.plot([x, x], [0, 80], color=PALETTE['ink'],
                             linewidth=1.5, zorder=4, linestyle='--')
            for y in Y_BINS[1:-1]:
                ax_grid.plot([0, 120], [y, y], color=PALETTE['ink'],
                             linewidth=1.5, zorder=4, linestyle='--')
            ax_grid.text(20, -6, 'Defensive\nThird', ha='center', fontsize=9,
                         color=PALETTE['ink'], fontweight='bold')
            ax_grid.text(60, -6, 'Middle\nThird', ha='center', fontsize=9,
                         color=PALETTE['ink'], fontweight='bold')
            ax_grid.text(100, -6, 'Attacking\nThird', ha='center', fontsize=9,
                         color=PALETTE['ink'], fontweight='bold')
            ax_grid.text(-10, 9, 'Left', ha='center', fontsize=9,
                         color=PALETTE['ink'], fontweight='bold', rotation=90)
            ax_grid.text(-10, 40, 'Central', ha='center', fontsize=9,
                         color=PALETTE['ink'], fontweight='bold', rotation=90)
            ax_grid.text(-10, 71, 'Right', ha='center', fontsize=9,
                         color=PALETTE['ink'], fontweight='bold', rotation=90)
            st.pyplot(fig_grid)
            plt.close()

        with col_pos:
            st.markdown(f"""
            <div class='info-card' style='margin-top:0;'>
            <table style='width:100%; font-size:0.85rem; border-collapse:collapse;'>
            <tr style='border-bottom:1px solid {PALETTE['rule']};'>
                <td style='padding:6px 8px; font-weight:700; color:{PALETTE['accent']};'>Defenders</td>
                <td style='padding:6px 8px;'>Centre Back, Left Back, Right Back</td>
            </tr>
            <tr style='border-bottom:1px solid {PALETTE['rule']};'>
                <td style='padding:6px 8px; font-weight:700; color:{PALETTE['accent']};'>Midfielders</td>
                <td style='padding:6px 8px;'>Defensive Midfield, Central Midfield, Left Midfield, Right Midfield</td>
            </tr>
            <tr>
                <td style='padding:6px 8px; font-weight:700; color:{PALETTE['accent']};'>Attackers</td>
                <td style='padding:6px 8px;'>Attacking Midfield, Left Wing, Right Wing, Centre Forward</td>
            </tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

    # ── TEAM-LEVEL EXPLANATION ──────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class='info-card'>

        <h4>1 — Match Scores</h4>
        <p>Each team's <b>match-level fluidity score</b> is the mean of all qualifying player z-scores recorded by
        that team in that match. After this aggregation, the team-match values are themselves z-standardised
        across the full cross-league distribution, producing a final score expressing how a team's match
        compared to the overall sample on a common scale.</p>

        <h4>2 — Seasonal Averages</h4>
        <p><b>Season-level team fluidity scores</b> are the mean of all of a team's match-level z-scores across
        the season. Positive values identify teams that played more fluidly than the overall sample average across
        the season; negative values identify teams more positional than the average.</p>

        <h4>3 — Tactical Consistency Score</h4>
        <p>A team's <b>tactical consistency</b> measures how reliably they adhered to a single tactical state
        (fluid or positional) across the season, calculated as:
        <code>(matches in dominant style / total matches) × 100</code>, where the dominant style is whichever of
        fluid (z &gt; 0) or positional (z &lt; 0) occurred most often. High consistency scores indicate teams
        with a clearly committed tactical identity; low scores indicate teams that switched between fluid and
        positional approaches match-to-match.</p>

        <h4>4 — Zonal Scores</h4>
        <p>To capture how fluidity is distributed across the pitch, position categories were grouped into three
        <b>zonal scores</b>: <b>Defence</b> (Centre Back, Full-Back), <b>Midfield</b> (Defensive, Central,
        Attacking, Wide Midfield) and <b>Attack</b> (Centre Forward, Wide Forward). For each zone, the mean of
        all qualifying player z-scores in that group was calculated, producing zonal scores at both match and
        season level. These reveal whether a team's fluidity is concentrated in a particular phase of the pitch
        or distributed evenly.</p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f"<div style='color:{PALETTE['ink_soft']}; font-size:0.95rem;'>"
        f"Select <b>Player Statistics</b> or <b>Team Statistics</b> from the sidebar to explore the data."
        f"</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PLAYER STATISTICS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Player Statistics":

    if not data_loaded:
        st.error("Data files not found.")
        st.stop()

    st.markdown("# Player Statistics")
    st.markdown("---")

    st.markdown('<div class="section-header">Select Player</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        leagues = sorted(player_season_stats['competition'].unique())
        selected_league = st.selectbox("League", leagues)

    with col2:
        league_positions = sorted(
            player_season_stats[player_season_stats['competition'] == selected_league]['position_cat'].unique()
        )
        selected_position = st.selectbox("Position Category", league_positions)

    with col3:
        pos_league_players = player_season_stats[
            (player_season_stats['competition'] == selected_league) &
            (player_season_stats['position_cat'] == selected_position)
        ].sort_values('season_fluidity', ascending=False)

        player_options = [
            f"{row['name']} ({row['team']}) z={row['season_fluidity']:.2f}"
            for _, row in pos_league_players.iterrows()
        ]
        selected_player_str = st.selectbox("Player", player_options)

    if not selected_player_str:
        st.stop()

    selected_player_name = selected_player_str.split(" (")[0]
    player_row = pos_league_players[pos_league_players['name'] == selected_player_name].iloc[0]
    player_id = player_row['player_id']

    st.markdown("---")

    tab_season, tab_match = st.tabs(["Season Stats", "Match Stats"])

    # ── SEASON STATS TAB ──────────────────────────────────────────────────
    with tab_season:

        st.markdown('<div class="section-header">Season Overview</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(stat_box("Team", player_row['team']), unsafe_allow_html=True)
        with c2:
            st.markdown(stat_box("Season Fluidity (z)", f"{player_row['season_fluidity']:.2f}"),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(stat_box("All-Leagues Rank", f"#{int(player_row['all_leagues_f_rank'])}"),
                        unsafe_allow_html=True)
        with c4:
            st.markdown(stat_box("League Rank", f"#{int(player_row['league_f_rank'])}"),
                        unsafe_allow_html=True)

        st.markdown("---")

        col_map, col_swarm = st.columns([1, 1])

        with col_map:
            st.markdown('<div class="section-header">Season Touch Locations</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:0.85rem; color:{PALETTE['ink_soft']}; margin-bottom:6px;'>"
                f"Attacking direction: left → right</div>",
                unsafe_allow_html=True
            )

            player_touches = load_touches(player_id)
            season_touches = player_touches[player_touches['position_norm'] == player_row['position_norm']]

            fig, ax = plt.subplots(figsize=(8, 5.5))
            fig.patch.set_facecolor(PALETTE['bg'])
            ax.set_facecolor(PALETTE['bg'])
            draw_binned_touch_map(season_touches, ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_swarm:
            st.markdown('<div class="section-header">Position Category Distribution</div>',
                        unsafe_allow_html=True)

            pos_data = player_season_stats[
                player_season_stats['position_cat'] == selected_position
            ].copy()

            fig, ax = plt.subplots(figsize=(6, 7))
            fig.patch.set_facecolor(PALETTE['bg'])
            ax.set_facecolor(PALETTE['bg'])

            other_players = pos_data[pos_data['name'] != selected_player_name]
            if len(other_players) > 0:
                sns.swarmplot(
                    data=other_players, x='position_cat', y='season_fluidity',
                    color=PALETTE['ink_muted'], size=6, alpha=0.55, ax=ax
                )

            player_pos_data = pos_data[pos_data['name'] == selected_player_name]
            if len(player_pos_data) > 0:
                ax.scatter(
                    [0], player_pos_data['season_fluidity'].values[0],
                    color=PALETTE['accent'], s=140, zorder=10,
                    edgecolors=PALETTE['ink'], linewidths=1.5,
                    label=selected_player_name
                )

            ax.axhline(y=0, color=PALETTE['ink'], linestyle='--',
                       linewidth=1, alpha=0.7,
                       label='Position avg (z = 0)')
            ax.set_xlabel(selected_position, fontsize=11, fontweight='bold')
            ax.set_ylabel('Season Fluidity (z-score)', fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.grid(axis='y', alpha=0.3, color=PALETTE['grid'])
            ax.legend(fontsize=9, loc='upper right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # Similar players (by z-score distance)
        st.markdown(
            f'<div class="section-header">Most Similar {selected_position}s (All Leagues)</div>',
            unsafe_allow_html=True
        )

        all_pos_players = player_season_stats[
            (player_season_stats['position_cat'] == selected_position) &
            (player_season_stats['name'] != selected_player_name)
        ].copy()
        all_pos_players['z_diff'] = abs(all_pos_players['season_fluidity'] - player_row['season_fluidity'])
        similar = all_pos_players.nsmallest(3, 'z_diff')[
            ['name', 'team', 'competition', 'season_fluidity']
        ]

        sc1, sc2, sc3 = st.columns(3)
        for i, (col, (_, row)) in enumerate(zip([sc1, sc2, sc3], similar.iterrows())):
            with col:
                st.markdown(f"""
                <div class='info-card' style='text-align:left;'>
                    <div style='font-size:0.7rem; font-weight:700; letter-spacing:0.06em;
                                text-transform:uppercase; color:{PALETTE['ink_muted']};
                                margin-bottom:4px;'>Rank #{i+1}</div>
                    <div style='font-size:1.05rem; font-weight:700; color:{PALETTE['ink']};
                                margin-bottom:2px;'>{row['name']}</div>
                    <div style='color:{PALETTE['ink_soft']}; font-size:0.85rem; margin-bottom:8px;'>
                        {row['team']} · {row['competition']}
                    </div>
                    <div style='font-size:1.4rem; font-weight:700;
                                color:{PALETTE['accent']};'>
                        z = {row['season_fluidity']:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── MATCH STATS TAB ────────────────────────────────────────────────────
    with tab_match:

        player_matches = player_match_stats[
            (player_match_stats['player_id'] == player_id) &
            (player_match_stats['position_norm'] == player_row['position_norm'])
        ].copy()

        if len(player_matches) == 0:
            st.info("No match data available.")
            st.stop()

        # Opponent lookup
        def get_opponent(match_id, team):
            same_match = team_match_stats[team_match_stats['match_id'] == match_id]
            opps = same_match[same_match['team'] != team]['team'].values
            return opps[0] if len(opps) > 0 else 'Unknown'

        ha_lookup = team_match_stats[['match_id', 'team', 'home_away']].drop_duplicates()
        player_matches = player_matches.merge(ha_lookup, on=['match_id', 'team'], how='left')
        player_matches['opponent'] = player_matches.apply(
            lambda r: get_opponent(r['match_id'], r['team']), axis=1
        )
        player_matches = player_matches.sort_values('fluidity_z', ascending=False).reset_index(drop=True)

        match_options = [
            f"vs {row['opponent']} ({row['home_away']}) z={row['fluidity_z']:.2f}"
            for _, row in player_matches.iterrows()
        ]
        match_id_map = {
            opt: row['match_id']
            for opt, (_, row) in zip(match_options, player_matches.iterrows())
        }

        st.markdown('<div class="section-header">Select Match</div>', unsafe_allow_html=True)
        selected_match_opt = st.selectbox("Match", match_options,
                                          key="player_match_select",
                                          label_visibility="collapsed")
        selected_match_id = match_id_map[selected_match_opt]
        match_row_pl = player_matches[player_matches['match_id'] == selected_match_id].iloc[0]

        st.markdown("---")
        st.markdown('<div class="section-header">Match Overview</div>', unsafe_allow_html=True)

        total_matches_pl = len(player_matches)
        match_rank_pl = int((player_matches['fluidity_z'] > match_row_pl['fluidity_z']).sum()) + 1

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(stat_box("Match Fluidity (z)", f"{match_row_pl['fluidity_z']:.2f}"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(stat_box("Season Rank", f"#{match_rank_pl} of {total_matches_pl} matches"),
                        unsafe_allow_html=True)

        st.markdown("---")

        # Match touch map + match-level position swarm side by side
        player_touches_all = load_touches(player_id)
        match_locs = player_touches_all[
            (player_touches_all['match_id'] == selected_match_id) &
            (player_touches_all['position_norm'] == player_row['position_norm'])
        ]
        n_touches = len(match_locs)

        col_mmap, col_mswarm = st.columns([1, 1])

        with col_mmap:
            st.markdown(
                f'<div class="section-header">Match Touch Locations · {n_touches} Touches</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size:0.85rem; color:{PALETTE['ink_soft']}; margin-bottom:6px;'>"
                f"Attacking direction: left → right</div>",
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(figsize=(8, 5.5))
            fig.patch.set_facecolor(PALETTE['bg'])
            ax.set_facecolor(PALETTE['bg'])
            draw_binned_touch_map(match_locs, ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_mswarm:
            st.markdown(
                '<div class="section-header">Position Category Match Distribution</div>',
                unsafe_allow_html=True
            )

            # All MATCH-level z scores within the player's position category
            pos_match_data = player_match_stats[
                player_match_stats['position_cat'] == selected_position
            ].copy()

            fig, ax = plt.subplots(figsize=(6, 7))
            fig.patch.set_facecolor(PALETTE['bg'])
            ax.set_facecolor(PALETTE['bg'])

            # Plot all other player-match combos as background
            mask_other = ~(
                (pos_match_data['player_id'] == player_id) &
                (pos_match_data['match_id'] == selected_match_id) &
                (pos_match_data['position_norm'] == player_row['position_norm'])
            )
            background = pos_match_data[mask_other]

            if len(background) > 0:
                sns.swarmplot(
                    data=background, x='position_cat', y='fluidity_z',
                    color=PALETTE['ink_muted'], size=3.5, alpha=0.45, ax=ax
                )

            # Highlight the selected player-position-match combination
            highlight = pos_match_data[
                (pos_match_data['player_id'] == player_id) &
                (pos_match_data['match_id'] == selected_match_id) &
                (pos_match_data['position_norm'] == player_row['position_norm'])
            ]
            if len(highlight) > 0:
                ax.scatter(
                    [0], highlight['fluidity_z'].values[0],
                    color=PALETTE['accent'], s=160, zorder=10,
                    edgecolors=PALETTE['ink'], linewidths=1.5,
                    label=f"{selected_player_name} (match)"
                )

            ax.axhline(y=0, color=PALETTE['ink'], linestyle='--',
                       linewidth=1, alpha=0.7,
                       label='Position avg (z = 0)')
            ax.set_xlabel(selected_position, fontsize=11, fontweight='bold')
            ax.set_ylabel('Match Fluidity (z-score)', fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.grid(axis='y', alpha=0.3, color=PALETTE['grid'])
            ax.legend(fontsize=9, loc='upper right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TEAM STATISTICS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Team Statistics":

    if not data_loaded:
        st.error("Data files not found.")
        st.stop()

    st.markdown("# Team Statistics")
    st.markdown("---")

    st.markdown('<div class="section-header">Select Team</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        leagues = sorted(team_season_stats['competition'].unique())
        selected_league = st.selectbox("League", leagues, key="team_league")
    with c2:
        league_teams_df = team_season_stats[
            team_season_stats['competition'] == selected_league
        ].sort_values('season_avg', ascending=False)
        league_teams = [
            f"{row['team']} (z={row['season_avg']:.2f})"
            for _, row in league_teams_df.iterrows()
        ]
        selected_team_str = st.selectbox("Team", league_teams, key="team_select")

    if not selected_team_str:
        st.stop()

    selected_team = selected_team_str.split(" (")[0]
    team_row = team_season_stats[team_season_stats['team'] == selected_team].iloc[0]
    team_matches = team_match_stats[team_match_stats['team'] == selected_team].copy()

    st.markdown("---")

    tab_team_season, tab_team_match = st.tabs(["Season Stats", "Match Stats"])

    # ── TEAM SEASON STATS ─────────────────────────────────────────────────
    with tab_team_season:

        st.markdown('<div class="section-header">Season Overview</div>', unsafe_allow_html=True)

        all_leagues_rank = int(
            (team_season_stats['season_avg'] > team_row['season_avg']).sum()
        ) + 1
        total_all = len(team_season_stats)
        league_rank = int(
            (team_season_stats[team_season_stats['competition'] == team_row['competition']]
             ['season_avg'] > team_row['season_avg']).sum()
        ) + 1
        total_league = len(
            team_season_stats[team_season_stats['competition'] == team_row['competition']]
        )

        # consistency score
        team_zscores = team_matches['team_match_z']
        consistency_count = max((team_zscores > 0).sum(), (team_zscores < 0).sum())
        total_matches_team = len(team_zscores)
        consistency_pct = (consistency_count / total_matches_team) * 100 if total_matches_team > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(stat_box("Season Avg Fluidity (z)", f"{team_row['season_avg']:.2f}"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(stat_box("All-Leagues Rank", f"#{all_leagues_rank} of {total_all}"),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(stat_box("League Rank", f"#{league_rank} of {total_league}"),
                        unsafe_allow_html=True)
        with c4:
            st.markdown(stat_box("Consistency", f"{consistency_pct:.0f}%"),
                        unsafe_allow_html=True)
        with c5:
            st.markdown(stat_box("Matches", int(team_row['matches'])),
                        unsafe_allow_html=True)

        st.markdown("---")

        # Zonal Fluidity Averages — vertical bar chart
        st.markdown('<div class="section-header">Zonal Fluidity Averages</div>',
                    unsafe_allow_html=True)

        zone_labels = list(ZONAL_COLS.keys())
        zone_values = [team_row.get(ZONAL_COLS[z], np.nan) for z in zone_labels]
        # bar colour by sign — fluid (green) vs positional (red)
        zone_colors = [PALETTE['fluid'] if (pd.notna(v) and v >= 0)
                       else PALETTE['positional'] for v in zone_values]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(PALETTE['bg'])
        ax.set_facecolor(PALETTE['bg'])

        x_pos = np.arange(len(zone_labels))
        bars = ax.bar(x_pos, zone_values, width=0.55,
                      color=zone_colors, edgecolor=PALETTE['ink'],
                      linewidth=0.8, zorder=3)

        # Value labels on top of each bar
        for bar, val in zip(bars, zone_values):
            if pd.notna(val):
                y_offset = 0.05 if val >= 0 else -0.12
                ax.text(bar.get_x() + bar.get_width()/2, val + y_offset,
                        f'{val:.2f}', ha='center',
                        va='bottom' if val >= 0 else 'top',
                        fontsize=11, fontweight='bold', color=PALETTE['ink'])

        ax.axhline(0, color=PALETTE['ink'], linewidth=1, linestyle='-', alpha=0.8, zorder=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(zone_labels, fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Fluidity (z-score)', fontsize=11, fontweight='bold')
        ax.set_ylim(min(min([v for v in zone_values if pd.notna(v)] + [0]) - 0.4, -1),
                    max(max([v for v in zone_values if pd.notna(v)] + [0]) + 0.4, 1))
        ax.grid(axis='y', alpha=0.3, color=PALETTE['grid'], zorder=0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")

        # Match Distribution line plot — coloured points + consistency in subtitle
        team_matches_sorted = team_matches.sort_values('match_id').reset_index(drop=True)
        team_matches_sorted['match_num'] = range(1, len(team_matches_sorted) + 1)

        # consistency components
        n_above = int((team_matches_sorted['team_match_z'] > 0).sum())
        n_below = int((team_matches_sorted['team_match_z'] < 0).sum())
        n_total = len(team_matches_sorted)
        dominant_count = max(n_above, n_below)
        consistency_calc = f"{dominant_count}/{n_total} × 100"
        season_avg_z = team_matches_sorted['team_match_z'].mean()

        st.markdown('<div class="section-header">Match Distribution</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:0.9rem; color:{PALETTE['ink_soft']}; margin-bottom:8px;'>"
            f"Consistency = <b style='color:{PALETTE['ink']};'>{consistency_pct:.0f}%</b> "
            f"({consistency_calc})</div>",
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.patch.set_facecolor(PALETTE['bg'])
        ax.set_facecolor(PALETTE['bg'])

        # Line
        ax.plot(team_matches_sorted['match_num'], team_matches_sorted['team_match_z'],
                color=PALETTE['ink_soft'], linewidth=1.2, zorder=2, alpha=0.7)

        # Coloured markers (fluid vs positional)
        fluid_mask = team_matches_sorted['team_match_z'] >= 0
        ax.scatter(team_matches_sorted.loc[fluid_mask, 'match_num'],
                   team_matches_sorted.loc[fluid_mask, 'team_match_z'],
                   color=PALETTE['fluid'], s=55, zorder=3,
                   edgecolors=PALETTE['ink'], linewidths=0.7, label='Fluid (z ≥ 0)')
        ax.scatter(team_matches_sorted.loc[~fluid_mask, 'match_num'],
                   team_matches_sorted.loc[~fluid_mask, 'team_match_z'],
                   color=PALETTE['positional'], s=55, zorder=3,
                   edgecolors=PALETTE['ink'], linewidths=0.7, label='Positional (z < 0)')

        ax.axhline(season_avg_z, color=PALETTE['accent'], linewidth=1.2, linestyle='--',
                   label=f'Season avg: {season_avg_z:.2f}', zorder=2)
        ax.axhline(0, color=PALETTE['ink'], linewidth=1, linestyle=':', alpha=0.5,
                   label='Cross-league avg (z = 0)', zorder=1)

        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel('Match Number (chronological)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Match Fluidity (z-score)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3, color=PALETTE['grid'], linestyle=':')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── TEAM MATCH STATS ──────────────────────────────────────────────────
    with tab_team_match:

        team_matches_with_meta = team_matches.copy()
        match_opponents = {}
        for _, row in team_matches_with_meta.iterrows():
            mid = row['match_id']
            same_match = team_match_stats[team_match_stats['match_id'] == mid]
            opponents = same_match[same_match['team'] != selected_team]['team'].values
            match_opponents[mid] = opponents[0] if len(opponents) > 0 else 'Unknown'
        team_matches_with_meta['opponent'] = team_matches_with_meta['match_id'].map(match_opponents)
        team_matches_sorted = team_matches_with_meta.sort_values('match_id').reset_index(drop=True)
        team_matches_sorted['match_num'] = range(1, len(team_matches_sorted) + 1)

        match_opts = {}
        for _, row in team_matches_sorted.iterrows():
            label = f"vs {row['opponent']} ({row['home_away']}) z={row['team_match_z']:.2f}"
            match_opts[label] = row['match_id']

        if not match_opts:
            st.info("No match data available.")
            st.stop()

        st.markdown('<div class="section-header">Select Match</div>', unsafe_allow_html=True)
        selected_match_label = st.selectbox(
            "Match", list(match_opts.keys()),
            key="team_match_select",
            label_visibility="collapsed"
        )
        selected_match_id = match_opts[selected_match_label]
        match_row = team_matches_sorted[team_matches_sorted['match_id'] == selected_match_id].iloc[0]

        st.markdown("---")

        # Comparison to Season — line plot with selected match highlighted
        st.markdown('<div class="section-header">Comparison to Season</div>', unsafe_allow_html=True)

        season_avg_z_t = team_matches_sorted['team_match_z'].mean()
        sel_match_row = team_matches_sorted[team_matches_sorted['match_id'] == selected_match_id]

        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.patch.set_facecolor(PALETTE['bg'])
        ax.set_facecolor(PALETTE['bg'])

        ax.plot(team_matches_sorted['match_num'], team_matches_sorted['team_match_z'],
                color=PALETTE['ink_soft'], linewidth=1.2, zorder=2, alpha=0.7)

        fluid_mask = team_matches_sorted['team_match_z'] >= 0
        ax.scatter(team_matches_sorted.loc[fluid_mask, 'match_num'],
                   team_matches_sorted.loc[fluid_mask, 'team_match_z'],
                   color=PALETTE['fluid'], s=50, zorder=3,
                   edgecolors=PALETTE['ink'], linewidths=0.7, label='Fluid (z ≥ 0)')
        ax.scatter(team_matches_sorted.loc[~fluid_mask, 'match_num'],
                   team_matches_sorted.loc[~fluid_mask, 'team_match_z'],
                   color=PALETTE['positional'], s=50, zorder=3,
                   edgecolors=PALETTE['ink'], linewidths=0.7, label='Positional (z < 0)')

        # Highlight the selected match
        if len(sel_match_row) > 0:
            ax.scatter(sel_match_row['match_num'], sel_match_row['team_match_z'],
                       color=PALETTE['accent'], s=180, zorder=5,
                       edgecolors=PALETTE['ink'], linewidths=1.4,
                       label='Selected Match', marker='D')

        ax.axhline(season_avg_z_t, color=PALETTE['accent'], linewidth=1.2, linestyle='--',
                   label=f'Season avg: {season_avg_z_t:.2f}', zorder=2)
        ax.axhline(0, color=PALETTE['ink'], linewidth=1, linestyle=':', alpha=0.5,
                   label='Cross-league avg (z = 0)', zorder=1)

        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel('Match Number (chronological)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Match Fluidity (z-score)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3, color=PALETTE['grid'], linestyle=':')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")

        # Zonal Match Averages — same structure as season but using match-level zonal cols
        st.markdown('<div class="section-header">Zonal Fluidity Averages (Match)</div>',
                    unsafe_allow_html=True)

        zone_labels = list(ZONAL_MATCH_COLS.keys())
        zone_values = [match_row.get(ZONAL_MATCH_COLS[z], np.nan) for z in zone_labels]
        zone_colors = [PALETTE['fluid'] if (pd.notna(v) and v >= 0)
                       else PALETTE['positional'] for v in zone_values]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(PALETTE['bg'])
        ax.set_facecolor(PALETTE['bg'])

        x_pos = np.arange(len(zone_labels))
        bars = ax.bar(x_pos, zone_values, width=0.55,
                      color=zone_colors, edgecolor=PALETTE['ink'],
                      linewidth=0.8, zorder=3)

        for bar, val in zip(bars, zone_values):
            if pd.notna(val):
                y_offset = 0.05 if val >= 0 else -0.12
                ax.text(bar.get_x() + bar.get_width()/2, val + y_offset,
                        f'{val:.2f}', ha='center',
                        va='bottom' if val >= 0 else 'top',
                        fontsize=11, fontweight='bold', color=PALETTE['ink'])

        ax.axhline(0, color=PALETTE['ink'], linewidth=1, linestyle='-', alpha=0.8, zorder=2)
        valid_vals = [v for v in zone_values if pd.notna(v)]
        if valid_vals:
            ax.set_ylim(min(min(valid_vals) - 0.4, -1),
                        max(max(valid_vals) + 0.4, 1))
        else:
            ax.set_ylim(-2, 2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(zone_labels, fontsize=11, fontweight='bold')
        ax.set_ylabel('Match Fluidity (z-score)', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, color=PALETTE['grid'], zorder=0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
