import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mplsoccer import Pitch
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Positional Fluidity Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8f9fa; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #eef2f7; }
    
    /* Metric boxes */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 10px 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="metric-container"] label {
        font-size: 0.75rem !important;
        color: #6c757d !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        color: #2c3e50 !important;
    }
    
    /* Section headers */
    .section-header {
        background-color: #e9ecef;
        border-left: 4px solid #4a7fb5;
        padding: 8px 14px;
        border-radius: 0 6px 6px 0;
        margin: 16px 0 12px 0;
        font-weight: 600;
        font-size: 1.05rem;
        color: #2c3e50;
    }

    /* Info card */
    .info-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        color: #2c3e50;
    }

    /* Nav buttons on home page */
    .nav-card {
        background: linear-gradient(135deg, #4a7fb5 0%, #2c5f8a 100%);
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        color: white;
        cursor: pointer;
        margin: 8px;
    }

    body, p, li, label { color: #2c3e50 !important; }
    h1 { color: #1a2e45 !important; }
    h2 { color: #2c3e50 !important; }
    h3 { color: #34495e !important; }
    h4 { color: #2c3e50 !important; }
    [data-testid="stSidebar"] * { color: #2c3e50 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #2c3e50 !important; }
    .stSelectbox label { font-weight: 600; color: #2c3e50 !important; }
    /* Selectbox container */
    div[data-baseweb="select"] { background-color: #ffffff !important; }
    div[data-baseweb="select"] > div { background-color: #ffffff !important; color: #2c3e50 !important; }
    div[data-baseweb="select"] * { color: #2c3e50 !important; }
    div[data-baseweb="select"] input { color: #2c3e50 !important; background-color: #ffffff !important; }
    div[data-baseweb="select"] [class*="ValueContainer"] { background-color: #ffffff !important; }
    div[data-baseweb="select"] [class*="SingleValue"] { color: #2c3e50 !important; }
    div[data-baseweb="select"] [class*="placeholder"] { color: #6c757d !important; }
    /* Dropdown popover */
    div[data-baseweb="popover"] { background-color: #ffffff !important; }
    div[data-baseweb="popover"] * { color: #2c3e50 !important; background-color: #ffffff !important; }
    div[data-baseweb="menu"] { background-color: #ffffff !important; }
    div[data-baseweb="menu"] li { color: #2c3e50 !important; background-color: #ffffff !important; }
    div[data-baseweb="menu"] li:hover { background-color: #eef2f7 !important; }
    /* Force all input elements */
    .stSelectbox input { color: #2c3e50 !important; background-color: #ffffff !important; }
    .stMarkdown p { color: #2c3e50 !important; }
    hr { border-color: #dee2e6; margin: 20px 0; }
    .val-positive { color: #2ecc71 !important; font-weight: 700 !important; font-size: 1.1rem !important; }
    .val-negative { color: #e74c3c !important; font-weight: 700 !important; font-size: 1.1rem !important; }
    .val-neutral   { color: #2c3e50 !important; font-weight: 700 !important; font-size: 1.1rem !important; }
    button[data-baseweb="tab"] { font-size: 1.05rem !important; font-weight: 600 !important; padding: 10px 20px !important; }
    button[data-baseweb="tab"] p { font-size: 1.05rem !important; color: #2c3e50 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ──────────────────────────────────────────────────────────────
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VALID_TOUCHES_GDRIVE_ID = "YOUR_FILE_ID_HERE"

@st.cache_data
def load_data():
    avg_team = pd.read_parquet(os.path.join(SCRIPT_DIR, "avg_team_stats.parquet"))
    team_match = pd.read_parquet(os.path.join(SCRIPT_DIR, "team_m_stats.parquet"))
    season_player = pd.read_parquet(os.path.join(SCRIPT_DIR, "player_s_stats.parquet"))
    filtered_pl = pd.read_parquet(os.path.join(SCRIPT_DIR, "included_players.parquet"))
    return avg_team, team_match, season_player, filtered_pl

@st.cache_data(show_spinner="Loading touch map data...")
def load_player_touches(player_id):
    import gdown
    output_path = "/tmp/included_touches.parquet"
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={VALID_TOUCHES_GDRIVE_ID}"
        gdown.download(url, output_path, quiet=False)
    touches = pd.read_parquet(output_path)
    return touches[touches['player_id'] == player_id]

try:
    avg_team_fluidity, team_match_stats, season_player_stats, filtered_players = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Data loading failed: {e}")

def stat_box(label, value, colour=None):
    css = f"class='{colour}'" if colour else "style='font-size:1.1rem; font-weight:600; color:#2c3e50;'"
    return (f"<div style='background:#ffffff; border:1px solid #dee2e6; border-radius:8px; "
            f"padding:10px 14px; box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
            f"<div style='font-size:0.72rem; color:#6c757d; margin-bottom:4px;'>{label}</div>"
            f"<div {css}>{value}</div>"
            f"</div>")

# ─── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Player Statistics", "Team Statistics"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.8rem; color:#6c757d; padding:8px;'>
<b>Data Source:</b> StatsBomb Open Data<br>
<b>Season:</b> 2015–16<br>
<b>Leagues:</b> Premier League · La Liga · Bundesliga · Serie A
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":

    st.markdown("# Positional Fluidity Dashboard: Team and Player Statistics")
    st.markdown("### Findings from the English Premier League, Spanish La Liga, German Bundesliga and Italian Serie A (2015–2016)")
    st.markdown("---")

    # ── What is positional fluidity? ──────────────────────────────────────────
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("## What is Positional Fluidity?")
        st.markdown("""
        <div class='info-card'>
        <p>
        Throughout football history, debates surrounding structure and freedom have fundamentally shaped tactical thinking. 
        This tension is characterised by the contrast between tactical systems championing <b>positional fluidity</b> — 
        where players have creative freedom to move widely across the pitch — and more organised, structured approaches 
        where players occupy defined zones and interact in pre-set patterns.
        </p>
        <p>
        For much of the twenty-first century, <b>positional play</b> (<i>juego de posición</i>) dominated global football, 
        driven by the success of Pep Guardiola at Barcelona, Bayern Munich, and Manchester City. However, recent years 
        have seen a resurgence of fluid, positionally expressive approaches. Carlo Ancelotti's Champions League winning 
        Real Madrid and Lionel Scaloni's World Cup winning Argentina both achieved success through positional freedom 
        in attacking movement.
        </p>
        <p>
        By 2023, these fluid approaches were unified under <b>relationism</b> — a direct counter-theory to positional play, 
        focused on creativity and movement, inspired by South American footballing traditions. Examples include 
        Fluminense (Copa Libertadores 2023) and NEC Nijmegen, who operate with almost no set positions during attacking 
        build-up. This debate between structure and fluidity represents one of the most compelling and under-examined 
        areas of modern football analytics.
        </p>
        <p>
        <b>Positional fluidity</b>, as measured in this project, refers to the degree to which a player covers more 
        spatial area than would typically be expected of their tactical position — operating in more diverse areas of 
        the pitch than their positional role would ordinarily dictate.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("## Why Does It Matter?")
        st.markdown("""
        <div class='info-card'>
        <p>
        Despite tactical systems being broadly recognised as an essential performance indicator in elite football, 
        <b>data-driven exploration of tactical behaviour</b> remains comparatively limited. Existing research has 
        been constrained by:
        </p>
        <ul>
            <li>Methodologically complex approaches (e.g. Voronoi cells, entropy indices) producing inaccessible outputs</li>
            <li>Reliance on expensive, non-public tracking data</li>
            <li>Limited direct analysis of the relationship between fluidity and attacking performance</li>
        </ul>
        <p>
        This project addresses these gaps by developing a <b>practical, event-data based fluidity metric</b> that 
        produces interpretable outputs for coaches, analysts, and fans alike — directly engaging with the 
        contemporary debate between structured positional play and relational fluidity.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── How Scores Were Calculated ────────────────────────────────────────────
    st.markdown("## How Were the Fluidity Scores Calculated?")

    tab1, tab2 = st.tabs(["Player-Level", "Team-Level"])

    with tab1:
        st.markdown("""
        <div class='info-card'>
        <h4>Step 1 — Touch Threshold</h4>
        <p>Ball touch events (passes, shots, dribbles, carries) were extracted from the StatsBomb dataset. 
        A minimum of <b>15 touches in a defined position per match</b> was required for inclusion, ensuring each 
        observation represents a meaningful positional contribution rather than a brief or incidental appearance.</p>
        
        <h4>Step 2 — Spatial Dispersion</h4>
        <p>For each qualifying player-position-match combination, <b>spatial dispersion</b> was calculated as the 
        product of the standard deviation of x-coordinates and y-coordinates across all qualifying touches: 
        <code>σ(x) · σ(y)</code>. This captures how widely spread a player's ball touch locations were across the pitch — 
        a simplified, axis-aligned adaptation of the area of the standard deviational ellipse. Higher values indicate 
        more spatially dispersed movement.</p>
        
        <h4>Step 3 — Position Category Z-Scoring</h4>
        <p>Raw spatial dispersion values differ substantially between positions by design — a centre back will 
        naturally have lower dispersion than an attacking midfielder. To account for this, dispersion values were 
        <b>z-score standardised within each position category</b>, producing fluidity scores that reflect how much 
        more (or less) spatial area a player covered relative to others in the same positional role.</p>
        
        <h4>Step 4 — Seasonal Aggregation</h4>
        <p>Players were required to meet the 15-touch threshold in at least <b>8 matches</b> in their primary 
        position to be included in seasonal analysis. Seasonal fluidity scores were calculated as the mean of all 
        qualifying match-level z-scores, then re-z-scored within position categories for comparative ranking.</p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class='info-card'>
        <h4>Step 1 — Team Match Scores</h4>
        <p>Team-level fluidity scores per match were calculated by averaging all qualifying player match z-scores 
        within the same team and match. Scores were also broken down by positional zone (Defensive, Midfield, Forward) 
        to enable zonal analysis. All match scores were then re-z-scored relative to the full cross-league distribution.</p>
        
        <h4>Step 2 — Seasonal Aggregation</h4>
        <p>Seasonal team fluidity scores were calculated as the mean of all match-level z-scores across the season. 
        These seasonal averages were re-z-scored relative to all 78 teams in the sample, enabling direct cross-league 
        comparison. Zonal seasonal scores (defensive, midfield, forward) were also re-z-scored independently.</p>
        
        <h4>Step 3 — Tactical Consistency</h4>
        <p>The standard deviation of a team's match-level fluidity z-scores was calculated as a measure of 
        <b>tactical consistency</b> — how variable a team's fluidity was from match to match across the season. 
        This was re-z-scored relative to all teams. Lower consistency scores indicate more predictable, stable 
        tactical behaviour; higher scores indicate greater match-to-match variation.</p>
        
        <h4>Step 4 — Tactical Profiles</h4>
        <p>Teams were assigned to one of four tactical profiles based on their seasonal fluidity and consistency scores:
        <b>Consistently Fluid</b>, <b>Inconsistently Fluid</b>, <b>Consistently Positional</b>, or 
        <b>Inconsistently Positional</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Navigation Cards ──────────────────────────────────────────────────────
    st.markdown("Select **Player Statistics** or **Team Statistics** from the sidebar to view match and season-level fluidity statistics and plots.")


# ══════════════════════════════════════════════════════════════════════════════
# PLAYER STATISTICS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Player Statistics":

    if not data_loaded:
        st.error("Data files not found. Please ensure CSV files are in the same directory as app.py.")
        st.stop()

    st.markdown("# Player Statistics")
    st.markdown("Select a player to view their positional fluidity profile, seasonal touch map, and similar players.")
    st.markdown("---")

    # ── Player Selection ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Player Selection</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        leagues = sorted(season_player_stats['competition'].unique())
        selected_league = st.selectbox("League", leagues)

    with col2:
        league_positions = sorted(
            season_player_stats[season_player_stats['competition'] == selected_league]['position_cat'].unique()
        )
        selected_position = st.selectbox("Position Category", league_positions)

    with col3:
        pos_league_players = season_player_stats[
            (season_player_stats['competition'] == selected_league) &
            (season_player_stats['position_cat'] == selected_position)
        ].sort_values('season_z', ascending=False)

        player_options = [
            f"{row['name']} ({row['team']}) z={row['season_z']:.2f}"
            for _, row in pos_league_players.iterrows()
        ]
        selected_player_str = st.selectbox("Player", player_options)

    if not selected_player_str:
        st.info("Please select a player above.")
        st.stop()

    # Extract player name and get data
    selected_player_name = selected_player_str.split(" (")[0]
    player_row = pos_league_players[pos_league_players['name'] == selected_player_name].iloc[0]

    st.markdown("---")

    player_id = player_row['player_id']

    # Get dominant position norm (used in both tabs)
    qualifying_norms = filtered_players[
        (filtered_players['player_id'] == player_id) &
        (filtered_players['position_cat'] == selected_position)
    ]
    dominant_norm = None
    all_norms = []
    if len(qualifying_norms) > 0:
        norm_apps = (
            qualifying_norms
            .groupby('position_norm')['match_id']
            .count()
            .reset_index(name='apps')
            .sort_values('apps', ascending=False)
        )
        dominant_norm = norm_apps.iloc[0]['position_norm']
        all_norms = qualifying_norms['position_norm'].unique()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_season, tab_match = st.tabs(["Season Stats", "Match Stats"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — SEASON STATS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_season:

        st.markdown(f'<div class="section-header">{selected_player_name} — Season Overview: {selected_position}</div>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(stat_box("Team", player_row['team']), unsafe_allow_html=True)
        with col2:
            st.markdown(stat_box("Seasonal Fluidity Score", f"{player_row['season_z']:.3f}"), unsafe_allow_html=True)
        with col3:
            st.markdown(stat_box("League Rank", f"#{int(player_row['league_fluidity_rank'])}"), unsafe_allow_html=True)
        with col4:
            st.markdown(stat_box("Global Rank", f"#{int(player_row['global_fluidity_rank'])}"), unsafe_allow_html=True)
        with col5:
            st.markdown(stat_box("Appearances", f"{int(player_row['apps'])}"), unsafe_allow_html=True)

        st.markdown("---")

        col_map, col_swarm = st.columns([1, 1])

        with col_map:
            st.markdown('<div class="section-header">Seasonal Touch Map</div>', unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.8rem; color:#6c757d; margin-bottom:6px;'>Attacking direction: left → right</div>", unsafe_allow_html=True)

            if dominant_norm is not None:
                player_touches = load_player_touches(player_id)
                player_locs = player_touches[
                    player_touches['position_norm'] == dominant_norm
                ]

                fig, ax = plt.subplots(figsize=(8, 5.5))
                fig.patch.set_facecolor('#f8f9fa')
                ax.set_facecolor('#f8f9fa')
                pitch = Pitch(pitch_type='statsbomb', line_color='#333333', pitch_color='#f8f9fa')
                pitch.draw(ax=ax)

                if len(player_locs) > 0:
                    pitch.kdeplot(
                        player_locs['x'], player_locs['y'],
                        ax=ax, cmap='Reds', fill=True,
                        bw_method=0.25, levels=12, thresh=0.02, alpha=0.8
                    )

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No touch data available for this player.")

        with col_swarm:
            st.markdown('<div class="section-header">Position Category Distribution</div>',
                        unsafe_allow_html=True)

            pos_data = season_player_stats[
                season_player_stats['position_cat'] == selected_position
            ].copy()

            fig, ax = plt.subplots(figsize=(6, 7))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')

            import seaborn as sns
            other_players = pos_data[pos_data['name'] != selected_player_name]
            if len(other_players) > 0:
                sns.swarmplot(
                    data=other_players, x='position_cat', y='season_z',
                    color='#adb5bd', size=6, alpha=0.6, ax=ax
                )

            player_pos_data = pos_data[pos_data['name'] == selected_player_name]
            if len(player_pos_data) > 0:
                ax.scatter(
                    [0], player_pos_data['season_z'].values[0],
                    color='#e74c3c', s=120, zorder=10,
                    edgecolors='black', linewidths=1.5
                )

            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
            ax.set_xlabel(selected_position, fontsize=10, fontweight='bold')
            ax.set_ylabel('Fluidity Z-Score', fontsize=10, fontweight='bold')
            ax.set_title(f'{selected_position} — Fluidity Distribution ({selected_player_name})', fontsize=10, fontweight='bold')
            ax.set_ylim(-5, 5)
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.grid(axis='y', alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # ── Similar Players ───────────────────────────────────────────────────
        st.markdown(f'<div class="section-header">Most Similar {selected_position}s (All Leagues)</div>',
                    unsafe_allow_html=True)

        all_pos_players = season_player_stats[
            (season_player_stats['position_cat'] == selected_position) &
            (season_player_stats['name'] != selected_player_name)
        ].copy()

        all_pos_players['z_diff'] = abs(all_pos_players['season_z'] - player_row['season_z'])
        similar = all_pos_players.nsmallest(3, 'z_diff')[
            ['name', 'team', 'competition', 'season_z', 'z_diff']
        ]

        col1, col2, col3 = st.columns(3)
        for i, (col, (_, row)) in enumerate(zip([col1, col2, col3], similar.iterrows())):
            with col:
                st.markdown(f"""
                <div class='info-card' style='text-align:center;'>
                    <div style='font-size:1.1rem; font-weight:700; color:#2c3e50;'>#{i+1} {row['name']}</div>
                    <div style='color:#6c757d; font-size:0.85rem;'>{row['team']} · {row['competition']}</div>
                    <div style='font-size:1.2rem; font-weight:700; color:#4a7fb5; margin-top:6px;'>
                        z = {row['season_z']:.3f}
                    </div>

                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — MATCH STATS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_match:

        st.markdown(f'<div class="section-header">{selected_player_name} ({selected_position}) — Match Stats</div>',
                    unsafe_allow_html=True)

        if dominant_norm is None:
            st.info("No match data available for this player.")
            st.stop()

        # Get all qualifying match-level observations for this player in this position cat
        # Use filtered_players which already passed the 8-match threshold at position_norm level
        player_match_data = filtered_players[
            (filtered_players['player_id'] == player_id) &
            (filtered_players['position_cat'] == selected_position)
        ][['match_id', 'position_norm', 'team', 'competition', 'match_z']].drop_duplicates(
            subset=['match_id', 'position_norm']
        ).copy()

        if len(player_match_data) == 0:
            st.info("No match-level data available for this player in this position.")
            st.stop()

        # Get home/away from team_match_stats which already has home_away column
        ha_lookup = team_match_stats[['match_id', 'team', 'home_away']].drop_duplicates()
        player_match_data = player_match_data.merge(ha_lookup, on=['match_id', 'team'], how='left')

        # Get opponent for each match
        def get_opponent(match_id, team):
            same_match = team_match_stats[team_match_stats['match_id'] == match_id]
            opps = same_match[same_match['team'] != team]['team'].values
            return opps[0] if len(opps) > 0 else 'Unknown'

        player_match_data['opponent'] = player_match_data.apply(
            lambda r: get_opponent(r['match_id'], r['team']), axis=1
        )
        player_match_data = player_match_data.sort_values('match_z', ascending=False).reset_index(drop=True)

        # Build dropdown options sorted by opponent with home/away and match z
        match_options = [
            f"vs {row['opponent']} ({row['home_away']}) z={row['match_z']:.2f}"
            for _, row in player_match_data.iterrows()
        ]
        match_id_map = {
            opt: (row['match_id'], row['position_norm'])
            for opt, (_, row) in zip(match_options, player_match_data.iterrows())
        }

        selected_match_opt = st.selectbox("Select Match", match_options, key="player_match_select")
        selected_match_id, selected_match_norm = match_id_map[selected_match_opt]
        match_row_pl = player_match_data[
            (player_match_data['match_id'] == selected_match_id) &
            (player_match_data['position_norm'] == selected_match_norm)
        ].iloc[0]

        st.markdown("---")

        col1, col2 = st.columns(2)
        total_matches_pl = len(player_match_data)
        match_rank_pl = int((player_match_data['match_z'] > match_row_pl['match_z']).sum()) + 1
        with col1:
            st.markdown(stat_box("Match Fluidity Score", f"{match_row_pl['match_z']:.3f}"), unsafe_allow_html=True)
        with col2:
            st.markdown(stat_box("Season Rank", f"#{match_rank_pl} of {total_matches_pl} matches"), unsafe_allow_html=True)

        st.markdown("---")

        # Match touch map
        player_touches = load_player_touches(player_id)
        match_locs = player_touches[
            (player_touches['match_id'] == selected_match_id) &
            (player_touches['position_norm'] == selected_match_norm)
        ]

        st.markdown(f'<div class="section-header">Match Touch Map ({len(match_locs)} touches)</div>', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.8rem; color:#6c757d; margin-bottom:6px;'>Attacking direction: left → right</div>", unsafe_allow_html=True)

        col_mmap, _ = st.columns([2, 1])
        with col_mmap:
            fig, ax = plt.subplots(figsize=(9, 6))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')
            pitch = Pitch(pitch_type='statsbomb', line_color='#333333', pitch_color='#f8f9fa')
            pitch.draw(ax=ax)

            if len(match_locs) > 0:
                pitch.scatter(
                    match_locs['x'], match_locs['y'],
                    ax=ax, s=50, color='#e74c3c', alpha=0.6, zorder=3,
                    edgecolors='white', linewidths=0.4
                )
            else:
                ax.text(60, 40, 'No touch data for this match',
                        ha='center', va='center', fontsize=11, color='#6c757d')

            opponent_name_pl = selected_match_opt.split("vs ")[1].split(" (")[0]
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()




# ══════════════════════════════════════════════════════════════════════════════
# TEAM STATISTICS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Team Statistics":

    if not data_loaded:
        st.error("Data files not found. Please ensure CSV files are in the same directory as app.py.")
        st.stop()

    st.markdown("# Team Statistics")
    st.markdown("Select a team to explore their seasonal fluidity profile, tactical positioning, and match-level performance.")
    st.markdown("---")

    # ── Team Selection ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Team Selection</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        leagues = sorted(avg_team_fluidity['competition'].unique())
        selected_league = st.selectbox("League", leagues, key="team_league")

    with col2:
        league_teams_df = avg_team_fluidity[
            avg_team_fluidity['competition'] == selected_league
        ].sort_values('season_z_score', ascending=False)
        league_teams = [
            f"{row['team']} (z={row['season_z_score']:.2f})"
            for _, row in league_teams_df.iterrows()
        ]
        selected_team_str = st.selectbox("Team", league_teams, key="team_select")

    if not selected_team_str:
        st.stop()

    selected_team = selected_team_str.split(" (z=")[0]

    team_row = avg_team_fluidity[avg_team_fluidity['team'] == selected_team].iloc[0]
    team_matches = team_match_stats[team_match_stats['team'] == selected_team].copy()

    st.markdown("---")

    tab_team_season, tab_team_match = st.tabs(["Season Stats", "Match Stats"])

    with tab_team_season:



        std_z = team_row['std_z_score']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(stat_box("Season Fluidity Z-Score", f"{team_row['season_z_score']:.3f}"), unsafe_allow_html=True)
        with col2:
            st.markdown(stat_box("Consistency Z-Score", f"{std_z:.3f}"), unsafe_allow_html=True)
        with col3:
            st.markdown(stat_box("Tactical Profile", team_row['tactical_profile']), unsafe_allow_html=True)
        with col4:
            st.markdown(stat_box("Matches", int(team_row['matches'])), unsafe_allow_html=True)

        st.markdown("---")

        col_quad, col_zone = st.columns([3, 2])

        with col_quad:
            st.markdown('<div class="section-header">Position in Tactical Profile Quadrant</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(8, 7))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')

            ax.fill_betweenx([-4, 0], -4, 0, color='lightblue', alpha=0.15)
            ax.fill_betweenx([0, 4], -4, 0, color='lightcoral', alpha=0.15)
            ax.fill_betweenx([-4, 0], 0, 4, color='lightgreen', alpha=0.15)
            ax.fill_betweenx([0, 4], 0, 4, color='lightyellow', alpha=0.25)

            ax.scatter(
                avg_team_fluidity['season_z_score'],
                avg_team_fluidity['std_z_score'],
                color='#adb5bd', edgecolors='white',
                linewidths=0.5, s=60, alpha=0.4
            )

            t_data = avg_team_fluidity[avg_team_fluidity['team'] == selected_team]
            ax.scatter(
                t_data['season_z_score'], t_data['std_z_score'],
                color='#e74c3c', edgecolors='black',
                linewidths=2, s=180, alpha=1.0, zorder=5
            )
            ax.annotate(
                selected_team,
                xy=(t_data['season_z_score'].values[0], t_data['std_z_score'].values[0]),
                xytext=(10, 8), textcoords='offset points',
                fontsize=9, fontweight='bold', color='#c0392b'
            )

            ax.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.4)
            ax.axhline(y=0, color='black', linewidth=1, linestyle='--', alpha=0.4)

            for txt, x, y, va in [
                ('Consistently\nPositional', -2, 3.5, 'top'),
                ('Consistently\nFluid', 2, 3.5, 'top'),
                ('Inconsistently\nPositional', -2, -3.5, 'bottom'),
                ('Inconsistently\nFluid', 2, -3.5, 'bottom'),
            ]:
                ax.text(x, y, txt, fontsize=9, va=va, ha='center',
                        color='#555', style='italic', fontweight='bold')

            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.grid(False)
            ax.set_xlabel('Fluidity Z-Score (Season Average)', fontsize=10)
            ax.set_ylabel('Consistency Z-Score (Season Std)', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_zone:
            st.markdown('<div class="section-header">Zonal Fluidity Breakdown</div>', unsafe_allow_html=True)

            zones = ['Defence', 'Midfield', 'Attack']
            values = [
                team_row['avg_defensive_score'],
                team_row['avg_midfield_score'],
                team_row['avg_forward_score']
            ]
            colours = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]

            fig, ax = plt.subplots(figsize=(5, 7))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')

            bars = ax.bar(zones, values, color=colours, edgecolor='black',
                          linewidth=0.8, width=0.5, alpha=0.85, zorder=3)

            ax.axhline(0, color='black', linewidth=1.2, linestyle='--',
                       alpha=0.5, label='Global Average')
            ax.set_ylabel('Fluidity Z-Score', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, frameon=False)
            ax.grid(axis='y', alpha=0.2, zorder=0)
            ax.set_ylim(-3, 3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        st.markdown('<div class="section-header">Match-by-Match Fluidity</div>', unsafe_allow_html=True)

        team_matches_sorted = team_matches.sort_values('match_id').reset_index(drop=True)
        team_matches_sorted['match_num'] = range(1, len(team_matches_sorted) + 1)

        match_opponents = {}
        for _, row in team_matches_sorted.iterrows():
            mid = row['match_id']
            same_match = team_match_stats[team_match_stats['match_id'] == mid]
            opponents = same_match[same_match['team'] != selected_team]['team'].values
            match_opponents[mid] = opponents[0] if len(opponents) > 0 else 'Unknown'
        team_matches_sorted['opponent'] = team_matches_sorted['match_id'].map(match_opponents)

        season_avg = team_matches_sorted['overall_z'].mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')
        ax.plot(team_matches_sorted['match_num'], team_matches_sorted['overall_z'],
                color='#4a7fb5', linewidth=1.5, marker='o', markersize=5, zorder=3)
        ax.axhline(season_avg, color='#e74c3c', linewidth=1.5, linestyle='--',
                   label=f'Season Avg: {season_avg:.2f}', zorder=2)
        ax.axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5,
                   label='Global Average', zorder=1)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlim(1, len(team_matches_sorted))
        ax.set_xlabel('Match Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fluidity Z-Score', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, frameon=False)
        ax.grid(True, alpha=0.15, linestyle=':')
        plt.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close()

    with tab_team_match:

        # Build team matches with opponent - home_away already in team_matches
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

        st.markdown('<div class="section-header">Select a Match</div>', unsafe_allow_html=True)

        match_opts = {}
        for _, row in team_matches_sorted.iterrows():
            mid = row['match_id']
            opp = row['opponent']
            ha = row['home_away']
            z = row['overall_z']
            label = f"vs {opp} ({ha}) z={z:.2f}"
            match_opts[label] = mid

        if not match_opts:
            st.info("No match data available for this team.")
            st.stop()

        selected_match_label = st.selectbox("Match", list(match_opts.keys()))
        selected_match_id = match_opts[selected_match_label]
        match_row = team_matches_sorted[team_matches_sorted['match_id'] == selected_match_id].iloc[0]

        st.markdown("---")

        st.markdown('<div class="section-header">Match Overview</div>', unsafe_allow_html=True)

        opponent_name = selected_match_label.split("vs ")[1].split(" (")[0]
        total_matches_tm = len(team_matches_sorted)
        match_rank_tm = int((team_matches_sorted['overall_z'] > match_row['overall_z']).sum()) + 1

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(stat_box("Match Fluidity Score", f"{match_row['overall_z']:.3f}"), unsafe_allow_html=True)
        with col2:
            st.markdown(stat_box("Season Rank (Fluidity Score)", f"{match_rank_tm} / {total_matches_tm} matches"), unsafe_allow_html=True)
        with col3:
            st.markdown(stat_box("Opponent", opponent_name), unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div class="section-header">Match Zonal Breakdown</div>', unsafe_allow_html=True)

        # Check for missing zonal data and warn
        zone_nulls = {
            'Defensive': pd.isna(match_row.get('defensive_z')),
            'Midfield': pd.isna(match_row.get('midfield_z')),
            'Attack': pd.isna(match_row.get('forward_z')),
        }
        missing_zones = [z for z, is_null in zone_nulls.items() if is_null]
        if missing_zones:
            missing_str = ', '.join(missing_zones).lower()
            st.warning(f"{missing_str.capitalize()} fluidity data missing for this match.")

        raw_match = [
            match_row['defensive_z'] if pd.notna(match_row.get('defensive_z')) else None,
            match_row['midfield_z'] if pd.notna(match_row.get('midfield_z')) else None,
            match_row['forward_z'] if pd.notna(match_row.get('forward_z')) else None,
        ]
        all_seasonal = [
            team_row['avg_defensive_score'],
            team_row['avg_midfield_score'],
            team_row['avg_forward_score']
        ]
        all_zones = ['Defence', 'Midfield', 'Attack']

        valid = [(z, m, s) for z, m, s in zip(all_zones, raw_match, all_seasonal) if m is not None]
        if valid:
            zones, match_values, seasonal_values = zip(*valid)
        else:
            zones, match_values, seasonal_values = [], [], []

        x = np.arange(len(zones))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')

        ax.bar(x - width / 2, match_values, width, label='Match',
               color='#4a7fb5', edgecolor='black', linewidth=0.7, alpha=0.9, zorder=3)
        ax.bar(x + width / 2, seasonal_values, width, label='Season Average',
               color='#adb5bd', edgecolor='black', linewidth=0.7, alpha=0.7, zorder=3)

        ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(zones, fontsize=11)
        ax.set_ylabel('Fluidity Score', fontsize=10)
        ax.set_ylim(-5, 5)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(axis='y', alpha=0.2, zorder=0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")
        st.markdown('<div class="section-header">Attacking Performance</div>', unsafe_allow_html=True)

        # Match-level xG and field tilt stat boxes
        col_xg, col_ft = st.columns(2)
        with col_xg:
            xg_val = match_row.get('shot_statsbomb_xg')
            st.markdown(stat_box("Match xG", f"{xg_val:.2f}" if pd.notna(xg_val) else "N/A"), unsafe_allow_html=True)
        with col_ft:
            ft_val = match_row.get('field_tilt')
            st.markdown(stat_box("Match Field Tilt %", f"{ft_val:.1f}" if pd.notna(ft_val) else "N/A"), unsafe_allow_html=True)

        st.markdown("---")

        # Line plots
        col_xg_line, col_ft_line = st.columns(2)

        with col_xg_line:
            st.markdown('<div class="section-header">xG — Season</div>', unsafe_allow_html=True)
            xg_data = team_matches_sorted.dropna(subset=['shot_statsbomb_xg']).copy()
            xg_season_avg = xg_data['shot_statsbomb_xg'].mean()
            xg_league_avg = team_match_stats[
                team_match_stats['competition'] == team_row['competition']
            ]['shot_statsbomb_xg'].mean()

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')

            ax.plot(xg_data['match_num'], xg_data['shot_statsbomb_xg'],
                    color='#adb5bd', linewidth=1.2, marker='o', markersize=4, zorder=2)
            ax.axhline(xg_season_avg, color='#e74c3c', linewidth=1.2, linestyle='--',
                       label=f'Team season avg: {xg_season_avg:.2f}', zorder=1)
            ax.axhline(xg_league_avg, color='#888888', linewidth=1.0, linestyle=':',
                       label=f'League avg: {xg_league_avg:.2f}', zorder=1)

            sel_xg_row = xg_data[xg_data['match_id'] == selected_match_id]
            if len(sel_xg_row) > 0:
                ax.scatter(sel_xg_row['match_num'], sel_xg_row['shot_statsbomb_xg'],
                           color='#e74c3c', s=100, zorder=5, edgecolors='black', linewidths=1.2)

            ax.set_xlabel('Match Number', fontsize=9)
            ax.set_ylabel('xG', fontsize=9)
            ax.set_ylim(0, 6)
            ax.legend(fontsize=8, frameon=False)
            ax.grid(True, alpha=0.15, linestyle=':')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_ft_line:
            st.markdown('<div class="section-header">Field Tilt % — Season</div>', unsafe_allow_html=True)
            ft_data = team_matches_sorted.dropna(subset=['field_tilt']).copy()
            ft_season_avg = ft_data['field_tilt'].mean()

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')

            ax.plot(ft_data['match_num'], ft_data['field_tilt'],
                    color='#adb5bd', linewidth=1.2, marker='o', markersize=4, zorder=2)
            ax.axhline(ft_season_avg, color='#e74c3c', linewidth=1.2, linestyle='--',
                       label=f'Team season avg: {ft_season_avg:.1f}%', zorder=1)
            ax.axhline(50, color='#888888', linewidth=1.0, linestyle=':',
                       label='League avg: 50%', zorder=1)

            sel_ft_row = ft_data[ft_data['match_id'] == selected_match_id]
            if len(sel_ft_row) > 0:
                ax.scatter(sel_ft_row['match_num'], sel_ft_row['field_tilt'],
                           color='#e74c3c', s=100, zorder=5, edgecolors='black', linewidths=1.2)

            ax.set_xlabel('Match Number', fontsize=9)
            ax.set_ylabel('Field Tilt %', fontsize=9)
            ax.set_ylim(0, 100)
            ax.legend(fontsize=8, frameon=False)
            ax.grid(True, alpha=0.15, linestyle=':')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

