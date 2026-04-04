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
    page_icon="⚽",
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
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
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

    h1 { color: #1a2e45; }
    h2 { color: #2c3e50; }
    h3 { color: #34495e; }
    
    .stSelectbox label { font-weight: 600; color: #2c3e50; }
    
    hr { border-color: #dee2e6; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ──────────────────────────────────────────────────────────────
VALID_TOUCHES_GDRIVE_ID = "YOUR_FILE_ID_HERE"

@st.cache_data
def load_data():
    import gdown
    avg_team = pd.read_csv("avg_team_scores.csv")
    team_match = pd.read_csv("team_match_scores.csv")
    season_player = pd.read_csv("player_stats_season.csv")
    filtered_pl = pd.read_csv("players_filtered.csv")
    url = f"https://drive.google.com/uc?id={VALID_TOUCHES_GDRIVE_ID}"
    output_path = "/tmp/valid_touches.csv"
    gdown.download(url, output_path, quiet=False)
    valid_t = pd.read_csv(output_path)
    return avg_team, team_match, season_player, filtered_pl, valid_t

try:
    avg_team_fluidity, team_match_stats, season_player_stats, filtered_players, valid_touches = load_data()
    data_loaded = True
except Exception:
    data_loaded = False

# ─── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚽ Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "👤 Player Statistics", "🏟️ Team Statistics"],
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
if page == "🏠 Home":

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

    tab1, tab2 = st.tabs(["👤 Player-Level", "🏟️ Team-Level"])

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
    st.markdown("## Explore the Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='nav-card'>
            <h2 style='color:white; margin:0;'>👤 Player Statistics</h2>
            <p style='color:#d0e4f7; margin:8px 0 0 0;'>
            Explore individual player fluidity scores, seasonal touch maps,
            position category rankings, and similar player comparisons.
            </p>
            <p style='color:#a8d4f5; margin:8px 0 0 0; font-size:0.9rem;'>
            ← Select <b>Player Statistics</b> from the sidebar to get started
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='nav-card'>
            <h2 style='color:white; margin:0;'>🏟️ Team Statistics</h2>
            <p style='color:#d0e4f7; margin:8px 0 0 0;'>
            Explore team tactical profiles, seasonal and match-level fluidity,
            zonal breakdowns, and tactical quadrant positioning.
            </p>
            <p style='color:#a8d4f5; margin:8px 0 0 0; font-size:0.9rem;'>
            ← Select <b>Team Statistics</b> from the sidebar to get started
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLAYER STATISTICS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Player Statistics":

    if not data_loaded:
        st.error("Data files not found. Please ensure CSV files are in the same directory as app.py.")
        st.stop()

    st.markdown("# 👤 Player Statistics")
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
            f"{row['player']} (z = {row['season_z']:.2f})"
            for _, row in pos_league_players.iterrows()
        ]
        selected_player_str = st.selectbox("Player", player_options)

    if not selected_player_str:
        st.info("Please select a player above.")
        st.stop()

    # Extract player name and get data
    selected_player_name = selected_player_str.split(" (z = ")[0]
    player_row = pos_league_players[pos_league_players['player'] == selected_player_name].iloc[0]

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
    tab_season, tab_match = st.tabs(["📅 Season Stats", "🎯 Match Stats"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — SEASON STATS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_season:

        st.markdown(f'<div class="section-header">{selected_player_name} ({selected_position}) — Season Overview</div>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Seasonal Fluidity Score", f"{player_row['season_z']:.3f}")
        with col2:
            st.metric("League Rank", f"#{int(player_row['league_fluidity_rank'])}")
        with col3:
            st.metric("Global Rank", f"#{int(player_row['global_fluidity_rank'])}")
        with col4:
            st.metric("Appearances", f"{int(player_row['apps'])}")

        st.markdown("---")

        col_map, col_swarm = st.columns([1, 1])

        with col_map:
            st.markdown('<div class="section-header">Seasonal Touch Map</div>', unsafe_allow_html=True)

            if dominant_norm is not None:
                player_locs = valid_touches[
                    (valid_touches['player_id'] == player_id) &
                    (valid_touches['position_norm'] == dominant_norm)
                ]

                fig, ax = plt.subplots(figsize=(8, 5.5))
                fig.patch.set_facecolor('#f8f9fa')
                ax.set_facecolor('#f8f9fa')
                pitch = Pitch(pitch_type='statsbomb', line_color='#333333', pitch_color='#f8f9fa')
                pitch.draw(ax=ax)

                if len(player_locs) > 0:
                    pitch.kdeplot(
                        player_locs['x'], player_locs['y'],
                        ax=ax, cmap='Blues', fill=True,
                        bw_method=0.3, levels=20, thresh=0.1, alpha=0.85
                    )

                norm_note = f" ({dominant_norm} only)" if len(all_norms) > 1 else ""
                ax.set_title(
                    f"{selected_player_name} — {selected_position}{norm_note}\n"
                    f"Season Fluidity Z-Score: {player_row['season_z']:.2f}",
                    fontsize=10, fontweight='bold', pad=8, color='#2c3e50'
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
            other_players = pos_data[pos_data['player'] != selected_player_name]
            if len(other_players) > 0:
                sns.swarmplot(
                    data=other_players, x='position_cat', y='season_z',
                    color='#adb5bd', size=6, alpha=0.6, ax=ax
                )

            player_pos_data = pos_data[pos_data['player'] == selected_player_name]
            if len(player_pos_data) > 0:
                ax.scatter(
                    [0], player_pos_data['season_z'].values[0],
                    color='#e74c3c', s=120, zorder=10,
                    edgecolors='black', linewidths=1.5
                )
                ax.annotate(
                    f"{selected_player_name.split()[0]} {selected_player_name.split()[-1]}\n"
                    f"({player_pos_data['season_z'].values[0]:.2f})",
                    xy=(0, player_pos_data['season_z'].values[0]),
                    xytext=(15, 0), textcoords='offset points',
                    fontsize=8.5, fontweight='bold', color='#e74c3c',
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2)
                )

            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
            ax.set_xlabel('')
            ax.set_ylabel('Fluidity Z-Score', fontsize=10, fontweight='bold')
            ax.set_title(f'{selected_position} — Fluidity Distribution', fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.grid(axis='y', alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # ── Similar Players ───────────────────────────────────────────────────
        st.markdown('<div class="section-header">Most Similar Players (All Leagues)</div>',
                    unsafe_allow_html=True)

        all_pos_players = season_player_stats[
            (season_player_stats['position_cat'] == selected_position) &
            (season_player_stats['player'] != selected_player_name)
        ].copy()

        all_pos_players['z_diff'] = abs(all_pos_players['season_z'] - player_row['season_z'])
        similar = all_pos_players.nsmallest(3, 'z_diff')[
            ['player', 'team', 'competition', 'season_z', 'z_diff']
        ]

        col1, col2, col3 = st.columns(3)
        for i, (col, (_, row)) in enumerate(zip([col1, col2, col3], similar.iterrows())):
            with col:
                st.markdown(f"""
                <div class='info-card' style='text-align:center;'>
                    <div style='font-size:1.1rem; font-weight:700; color:#2c3e50;'>#{i+1} {row['player']}</div>
                    <div style='color:#6c757d; font-size:0.85rem;'>{row['team']} · {row['competition']}</div>
                    <div style='font-size:1.2rem; font-weight:700; color:#4a7fb5; margin-top:6px;'>
                        z = {row['season_z']:.3f}
                    </div>
                    <div style='font-size:0.8rem; color:#adb5bd;'>Δ {row['z_diff']:.3f} from selected</div>
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

        # Get opponent for each match from team_match_stats
        def get_opponent(match_id, team):
            same_match = team_match_stats[team_match_stats['match_id'] == match_id]
            opps = same_match[same_match['team'] != team]['team'].values
            return opps[0] if len(opps) > 0 else 'Unknown'

        player_match_data['opponent'] = player_match_data.apply(
            lambda r: get_opponent(r['match_id'], r['team']), axis=1
        )
        player_match_data = player_match_data.sort_values('match_id').reset_index(drop=True)

        # Build dropdown options
        match_options = [
            f"vs {row['opponent']}  [{row['position_norm']}]  (z = {row['match_z']:.3f})"
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

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Match Fluidity Score", f"{match_row_pl['match_z']:.3f}")
        with col2:
            season_avg_pl = player_match_data['match_z'].mean()
            delta_pl = match_row_pl['match_z'] - season_avg_pl
            st.metric("Difference from Season Average", f"{delta_pl:+.3f}")
        with col3:
            st.metric("Position (this match)", match_row_pl['position_norm'])

        st.markdown("---")

        # Match touch map
        st.markdown('<div class="section-header">Match Touch Map</div>', unsafe_allow_html=True)

        match_locs = valid_touches[
            (valid_touches['player_id'] == player_id) &
            (valid_touches['match_id'] == selected_match_id) &
            (valid_touches['position_norm'] == selected_match_norm)
        ]

        col_mmap, col_minfo = st.columns([2, 1])
        with col_mmap:
            fig, ax = plt.subplots(figsize=(9, 6))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')
            pitch = Pitch(pitch_type='statsbomb', line_color='#333333', pitch_color='#f8f9fa')
            pitch.draw(ax=ax)

            if len(match_locs) >= 3:
                pitch.kdeplot(
                    match_locs['x'], match_locs['y'],
                    ax=ax, cmap='Reds', fill=True,
                    bw_method=0.4, levels=15, thresh=0.1, alpha=0.85
                )
            elif len(match_locs) > 0:
                pitch.scatter(
                    match_locs['x'], match_locs['y'],
                    ax=ax, s=60, color='#e74c3c', alpha=0.7, zorder=3
                )
            else:
                ax.text(60, 40, 'No touch data for this match',
                        ha='center', va='center', fontsize=11, color='#6c757d')

            opponent_name_pl = selected_match_opt.split("vs ")[1].split("  ")[0]
            ax.set_title(
                f"{selected_player_name} vs {opponent_name_pl} — {selected_match_norm}\n"
                f"Match Fluidity Z-Score: {match_row_pl['match_z']:.3f}  |  "
                f"Touches: {len(match_locs)}",
                fontsize=10, fontweight='bold', pad=8, color='#2c3e50'
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_minfo:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown(f"**{selected_player_name}**")
            st.markdown(f"*vs {opponent_name_pl}*")
            st.markdown("---")
            st.markdown(f"**Position:** {match_row_pl['position_norm']}")
            st.markdown(f"**Match Fluidity:** `{match_row_pl['match_z']:.3f}`")
            st.markdown(f"**Season Average:** `{season_avg_pl:.3f}`")
            delta_col_pl = "#2ecc71" if delta_pl > 0 else "#e74c3c"
            sign_pl = "+" if delta_pl > 0 else ""
            st.markdown(
                f"**Δ from Avg:** <span style='color:{delta_col_pl}; font-weight:700;'>"
                f"{sign_pl}{delta_pl:.3f}</span>",
                unsafe_allow_html=True
            )
            st.markdown(f"**Touches recorded:** {len(match_locs)}")
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TEAM STATISTICS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏟️ Team Statistics":

    if not data_loaded:
        st.error("Data files not found. Please ensure CSV files are in the same directory as app.py.")
        st.stop()

    st.markdown("# 🏟️ Team Statistics")
    st.markdown("Select a team to explore their seasonal fluidity profile, tactical positioning, and match-level performance.")
    st.markdown("---")

    # ── Team Selection ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Team Selection</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        leagues = sorted(avg_team_fluidity['competition'].unique())
        selected_league = st.selectbox("League", leagues, key="team_league")

    with col2:
        league_teams = sorted(
            avg_team_fluidity[avg_team_fluidity['competition'] == selected_league]['team'].unique()
        )
        selected_team = st.selectbox("Team", league_teams, key="team_select")

    if not selected_team:
        st.stop()

    team_row = avg_team_fluidity[avg_team_fluidity['team'] == selected_team].iloc[0]
    team_matches = team_match_stats[team_match_stats['team'] == selected_team].copy()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — SEASON STATS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📊 Season Stats")

    # ── Key Stats ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Season Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Season Fluidity Z-Score", f"{team_row['season_z_score']:.3f}")
    with col2:
        st.metric("Consistency Z-Score", f"{team_row['std_z_score']:.3f}")
    with col3:
        st.metric("Tactical Profile", team_row['tactical_profile'])
    with col4:
        st.metric("Matches Included", int(team_row['matches_included']))

    st.markdown("---")

    # ── Quadrant + Zonal Bar ──────────────────────────────────────────────────
    col_quad, col_zone = st.columns([3, 2])

    with col_quad:
        st.markdown('<div class="section-header">Tactical Profile Quadrant</div>', unsafe_allow_html=True)

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
            ('Inconsistently\nPositional', -2, 3.5, 'top'),
            ('Inconsistently\nFluid', 2, 3.5, 'top'),
            ('Consistently\nPositional', -2, -3.5, 'bottom'),
            ('Consistently\nFluid', 2, -3.5, 'bottom'),
        ]:
            ax.text(x, y, txt, fontsize=9, va=va, ha='center',
                    color='#555', style='italic', fontweight='bold')

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(False)
        ax.set_xlabel('Fluidity Z-Score (Season Average)', fontsize=10)
        ax.set_ylabel('Consistency Z-Score (Season Std)', fontsize=10)
        ax.set_title('Tactical Profile — All Teams', fontsize=11, fontweight='bold')
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

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')

        bars = ax.bar(zones, values, color=colours, edgecolor='black',
                      linewidth=0.8, width=0.5, alpha=0.85, zorder=3)

        for bar, val in zip(bars, values):
            y_pos = val + 0.05 if val >= 0 else val - 0.05
            va = 'bottom' if val >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:+.2f}', ha='center', va=va,
                    fontsize=9, fontweight='bold')

        ax.axhline(0, color='black', linewidth=1.2, linestyle='--',
                   alpha=0.5, label='Global Average')
        ax.set_ylabel('Fluidity Z-Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{selected_team}\nSeasonal Zonal Fluidity', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis='y', alpha=0.2, zorder=0)
        ax.set_ylim(-3, 3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Match-by-Match Line Chart ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Match-by-Match Fluidity</div>', unsafe_allow_html=True)

    team_matches_sorted = team_matches.sort_values('match_id').reset_index(drop=True)
    team_matches_sorted['match_num'] = range(1, len(team_matches_sorted) + 1)

    # Hover tooltip using plotly
    try:
        import plotly.graph_objects as go

        # Get opponent info from match_sts (the attacking stats table which has opponent)
        # We need to reconstruct opponent from team_match_stats
        # Each match has two teams — find the other one
        match_opponents = {}
        for _, row in team_matches_sorted.iterrows():
            mid = row['match_id']
            same_match = team_match_stats[team_match_stats['match_id'] == mid]
            opponents = same_match[same_match['team'] != selected_team]['team'].values
            match_opponents[mid] = opponents[0] if len(opponents) > 0 else 'Unknown'

        team_matches_sorted['opponent'] = team_matches_sorted['match_id'].map(match_opponents)

        hover_text = [
            f"Match {row['match_num']}<br>vs {row['opponent']}<br>Fluidity: {row['overall_z']:.3f}"
            for _, row in team_matches_sorted.iterrows()
        ]

        season_avg = team_matches_sorted['overall_z'].mean()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=team_matches_sorted['match_num'],
            y=team_matches_sorted['overall_z'],
            mode='lines+markers',
            name='Match Fluidity',
            line=dict(color='#4a7fb5', width=2),
            marker=dict(size=6, color='#4a7fb5'),
            hovertext=hover_text,
            hoverinfo='text'
        ))

        fig.add_hline(y=season_avg, line_dash='dash', line_color='#e74c3c',
                      annotation_text=f"Season Avg: {season_avg:.2f}",
                      annotation_position='top right')
        fig.add_hline(y=0, line_dash='dot', line_color='black',
                      annotation_text="Global Avg",
                      annotation_position='bottom right',
                      line=dict(color='black', width=1, dash='dot'))

        fig.update_layout(
            title=f"{selected_team} — Match-by-Match Fluidity (2015–16)",
            xaxis_title="Match Number",
            yaxis_title="Fluidity Z-Score",
            yaxis=dict(range=[-4.5, 4.5]),
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa',
            font=dict(family='Arial', size=11),
            legend=dict(orientation='h', y=-0.15),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback to matplotlib
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')

        ax.plot(team_matches_sorted['match_num'], team_matches_sorted['overall_z'],
                color='#4a7fb5', linewidth=1.5, marker='o', markersize=5, zorder=3)
        ax.axhline(team_matches_sorted['overall_z'].mean(), color='#e74c3c',
                   linewidth=1.5, linestyle='--', label='Season Average')
        ax.axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5, label='Global Average')
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel('Match Number', fontsize=11)
        ax.set_ylabel('Fluidity Z-Score', fontsize=11)
        ax.set_title(f'{selected_team} — Match-by-Match Fluidity', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — MATCH STATS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🎯 Match Stats")
    st.markdown('<div class="section-header">Select a Match</div>', unsafe_allow_html=True)

    # Build match dropdown options
    match_opts = {}
    for _, row in team_matches_sorted.iterrows():
        mid = row['match_id']
        opp = match_opponents.get(mid, 'Unknown') if 'match_opponents' in dir() else 'Unknown'
        label = f"vs {opp}  (Fluidity: {row['overall_z']:.3f})"
        match_opts[label] = mid

    if not match_opts:
        st.info("No match data available for this team.")
        st.stop()

    selected_match_label = st.selectbox("Match", list(match_opts.keys()))
    selected_match_id = match_opts[selected_match_label]
    match_row = team_matches_sorted[team_matches_sorted['match_id'] == selected_match_id].iloc[0]

    st.markdown("---")

    # ── Match Key Stats ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Match Overview</div>', unsafe_allow_html=True)

    season_avg_score = team_matches_sorted['overall_z'].mean()
    diff_from_avg = match_row['overall_z'] - season_avg_score

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Match Fluidity Score", f"{match_row['overall_z']:.3f}")
    with col2:
        delta_str = f"{diff_from_avg:+.3f} vs season avg"
        st.metric("Difference from Season Average", f"{diff_from_avg:+.3f}",
                  delta=delta_str)
    with col3:
        opponent_name = selected_match_label.split("vs ")[1].split("  ")[0]
        st.metric("Opponent", opponent_name)

    st.markdown("---")

    # ── Match Zonal Bar Chart ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Match Zonal Fluidity Breakdown</div>', unsafe_allow_html=True)

    col_mzone, col_minfo = st.columns([2, 1])

    with col_mzone:
        match_values = [
            match_row.get('defensive_z', 0),
            match_row.get('midfield_z', 0),
            match_row.get('forward_z', 0)
        ]
        seasonal_values = [
            team_row['avg_defensive_score'],
            team_row['avg_midfield_score'],
            team_row['avg_forward_score']
        ]
        zones = ['Defence', 'Midfield', 'Attack']
        x = np.arange(len(zones))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')

        m_colours = ['#2ecc71' if v >= 0 else '#e74c3c' for v in match_values]
        s_colours = ['#85c1e9' if v >= 0 else '#f1948a' for v in seasonal_values]

        bars1 = ax.bar(x - width / 2, match_values, width, label='This Match',
                       color=m_colours, edgecolor='black', linewidth=0.7, alpha=0.9, zorder=3)
        bars2 = ax.bar(x + width / 2, seasonal_values, width, label='Season Average',
                       color=s_colours, edgecolor='black', linewidth=0.7, alpha=0.7, zorder=3)

        for bar, val in zip(bars1, match_values):
            y_pos = val + 0.04 if val >= 0 else val - 0.04
            va = 'bottom' if val >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:+.2f}', ha='center', va=va, fontsize=8, fontweight='bold')

        ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(zones, fontsize=10)
        ax.set_ylabel('Fluidity Z-Score', fontsize=10)
        ax.set_title(f'Zonal Fluidity — This Match vs Season Average', fontsize=10, fontweight='bold')
        ax.legend(fontsize=9, frameon=False)
        ax.grid(axis='y', alpha=0.2, zorder=0)
        ax.set_ylim(-3, 3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_minfo:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown(f"**Match vs {opponent_name}**")
        st.markdown("---")
        st.markdown(f"**Overall Fluidity:** `{match_row['overall_z']:.3f}`")
        st.markdown(f"**Defensive Zone:** `{match_row.get('defensive_z', 0):.3f}`")
        st.markdown(f"**Midfield Zone:** `{match_row.get('midfield_z', 0):.3f}`")
        st.markdown(f"**Forward Zone:** `{match_row.get('forward_z', 0):.3f}`")
        st.markdown("---")
        st.markdown(f"**Season Avg:** `{season_avg_score:.3f}`")
        delta_col = "#2ecc71" if diff_from_avg > 0 else "#e74c3c"
        sign = "+" if diff_from_avg > 0 else ""
        st.markdown(
            f"**Δ from Avg:** <span style='color:{delta_col}; font-weight:700;'>{sign}{diff_from_avg:.3f}</span>",
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
