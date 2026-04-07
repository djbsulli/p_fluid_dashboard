import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mplsoccer import Pitch
import warnings
warnings.filterwarnings('ignore')
import os
import seaborn as sns

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
    .stApp { background-color: #f8f9fa; }
    [data-testid="stSidebar"] { background-color: #eef2f7; }
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 10px 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
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
    .info-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        color: #2c3e50;
    }
    body, p, li, label { color: #2c3e50 !important; }
    h1 { color: #1a2e45 !important; }
    h2 { color: #2c3e50 !important; }
    h3 { color: #34495e !important; }
    [data-testid="stSidebar"] * { color: #2c3e50 !important; }
    .stSelectbox label { font-weight: 600; color: #2c3e50 !important; }
    div[data-baseweb="select"] { background-color: #ffffff !important; }
    div[data-baseweb="select"] > div { background-color: #ffffff !important; color: #2c3e50 !important; }
    div[data-baseweb="select"] * { color: #2c3e50 !important; }
    .stMarkdown p { color: #2c3e50 !important; }
    hr { border-color: #dee2e6; margin: 20px 0; }
    button[data-baseweb="tab"] { font-size: 1.05rem !important; font-weight: 600 !important; padding: 10px 20px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VALID_TOUCHES_GDRIVE_ID = "1EGNoqoQJXe8aOjFe5BUw2ya41Yj6eBYg"

@st.cache_data
def load_data():
    season_player  = pd.read_parquet(os.path.join(SCRIPT_DIR, "pf_player_season.parquet"))
    team_match     = pd.read_parquet(os.path.join(SCRIPT_DIR, "pf_team_match.parquet"))
    avg_team       = pd.read_parquet(os.path.join(SCRIPT_DIR, "pf_team_season.parquet"))
    concentration  = pd.read_parquet(os.path.join(SCRIPT_DIR, "pf_player_match.parquet"))
    return season_player, team_match, avg_team, concentration

@st.cache_data(show_spinner="Loading touch data...")
def load_touches(player_id):
    import gdown
    output_path = "/tmp/pf_touches.parquet"
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={VALID_TOUCHES_GDRIVE_ID}"
        gdown.download(url, output_path, quiet=False)
    touches = pd.read_parquet(output_path)
    return touches[touches['player_id'] == player_id]

try:
    season_player_stats, team_match_stats, avg_team_fluidity, player_match_stats = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Data loading failed: {e}")

# ─── Zone definitions ──────────────────────────────────────────────────────────
x_bins_9 = [0, 40, 80, 120]
y_bins_9 = [0, 26.7, 53.3, 80]

def assign_zone_9(x, y):
    x_zone = np.digitize(np.array(x), x_bins_9) - 1
    y_zone = np.digitize(np.array(y), y_bins_9) - 1
    x_zone = np.clip(x_zone, 0, len(x_bins_9) - 2)
    y_zone = np.clip(y_zone, 0, len(y_bins_9) - 2)
    return x_zone * 3 + y_zone + 1

# ─── Stat box helper ───────────────────────────────────────────────────────────
def stat_box(label, value):
    return (f"<div style='background:#ffffff; border:1px solid #dee2e6; border-radius:8px; "
            f"padding:10px 14px; box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
            f"<div style='font-size:0.72rem; color:#6c757d; margin-bottom:4px;'>{label}</div>"
            f"<div style='font-size:1.1rem; font-weight:600; color:#2c3e50;'>{value}</div>"
            f"</div>")

# ─── Touch map helper ─────────────────────────────────────────────────────────
def draw_binned_touch_map(touch_df, ax, title_suffix=""):
    pitch = Pitch(pitch_type='statsbomb', line_color='#333333', pitch_color='#f8f9fa')
    pitch.draw(ax=ax)

    if len(touch_df) == 0:
        ax.text(60, 40, 'No touch data', ha='center', va='center',
                fontsize=11, color='#6c757d')
        return

    touch_df = touch_df.copy()
    touch_df['zone_9'] = assign_zone_9(touch_df['x'].values, touch_df['y'].values)
    total = len(touch_df)
    zone_pcts = touch_df['zone_9'].value_counts(normalize=True) * 100

    for zone_num in range(1, 10):
        col = (zone_num - 1) // 3
        row = (zone_num - 1) % 3
        x0, x1 = x_bins_9[col], x_bins_9[col + 1]
        y0, y1 = y_bins_9[row], y_bins_9[row + 1]
        pct = zone_pcts.get(zone_num, 0)
        color = plt.cm.Reds(min(pct / 40, 1.0))
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              color=color, zorder=2, alpha=0.75)
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f'{pct:.0f}%',
                ha='center', va='center', fontsize=13,
                fontweight='bold', color='black', zorder=4)

    # grid lines
    for x in x_bins_9[1:-1]:
        ax.plot([x, x], [0, 80], color='white', linewidth=1.2,
                zorder=3, linestyle='--', alpha=0.7)
    for y in y_bins_9[1:-1]:
        ax.plot([0, 120], [y, y], color='white', linewidth=1.2,
                zorder=3, linestyle='--', alpha=0.7)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Select Page",
                         ["Home", "Player Statistics", "Team Statistics"],
                         label_visibility="collapsed")
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

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("## What is Positional Fluidity?")
        st.markdown("""
        <div class='info-card'>
        <p>Throughout football history, debates surrounding structure and freedom have fundamentally shaped tactical thinking.
        This tension is characterised by the contrast between tactical systems championing <b>positional fluidity</b> —
        where players have creative freedom to move widely across the pitch — and more organised, structured approaches
        where players occupy defined zones and interact in pre-set patterns.</p>
        <p>For much of the twenty-first century, <b>positional play</b> (<i>juego de posición</i>) dominated global football,
        driven by the success of Pep Guardiola at Barcelona, Bayern Munich, and Manchester City. However, recent years
        have seen a resurgence of fluid, positionally expressive approaches.</p>
        <p><b>Positional fluidity</b>, as measured in this project, refers to the degree to which a player operates
        across multiple areas of the pitch rather than concentrating their activity in a single consistent zone —
        expressed as the percentage of touches occurring outside a player's most frequently occupied pitch zone.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("## Why Does It Matter?")
        st.markdown("""
        <div class='info-card'>
        <p>Despite tactical systems being broadly recognised as an essential performance indicator in elite football,
        <b>data-driven exploration of tactical behaviour</b> remains comparatively limited.</p>
        <p>This project addresses these gaps by developing a <b>practical, event-data based fluidity metric</b> that
        produces interpretable percentage outputs for coaches, analysts, and fans alike.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## How Were the Fluidity Scores Calculated?")

    tab1, tab2 = st.tabs(["Player-Level", "Team-Level"])

    with tab1:
        st.markdown("""
        <div class='info-card'>
        <h4>Step 1 — Touch Threshold</h4>
        <p>Ball touch events were extracted from the StatsBomb dataset. A minimum of <b>15 touches in a defined
        position per match</b> was required for inclusion.</p>
        <h4>Step 2 — Zone Assignment</h4>
        <p>The pitch was divided into a <b>9-zone grid</b> (3 columns × 3 rows) aligned with the Juego de Posición
        spatial framework (Rowlinson, 2021). Each touch was assigned to one of the 9 zones based on its x/y coordinates.</p>
        <h4>Step 3 — Fluidity Percentage</h4>
        <p>For each qualifying player-position-match combination, the <b>fluidity score</b> was calculated as the
        percentage of touches falling outside the player's most frequently occupied zone:
        <code>(1 - touches in modal zone / total touches) × 100</code>. Higher scores indicate more spatially
        diverse movement.</p>
        <h4>Step 4 — Seasonal Aggregation</h4>
        <p>Players were required to meet the 15-touch threshold in at least <b>8 matches</b> in their primary
        position to be included in seasonal analysis. Seasonal fluidity scores were calculated as the mean of all
        qualifying match-level scores.</p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class='info-card'>
        <h4>Step 1 — Player Z-Scoring</h4>
        <p>Match-level fluidity percentages were z-scored within position category to remove the natural positional
        gradient — full-backs score lower than strikers by design. This ensures the team score reflects tactical
        fluidity rather than positional composition.</p>
        <h4>Step 2 — Team Match Score</h4>
        <p>Team-level fluidity scores per match were calculated by averaging all qualifying player z-scores within
        the same team and match. Scores were also broken down by positional zone (Defensive, Midfield, Forward).
        Match scores were re-z-scored relative to the full cross-league distribution.</p>
        <h4>Step 3 — Seasonal Aggregation</h4>
        <p>Seasonal team fluidity scores were calculated as the mean of all match-level z-scores, then re-z-scored
        relative to all 78 teams in the sample.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("Select **Player Statistics** or **Team Statistics** from the sidebar to explore the data.")


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
        ].sort_values('season_fluidity', ascending=False)

        player_options = [
            f"{row['name']} ({row['team']}) {row['season_fluidity']:.1f}%"
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

    # ── SEASON STATS TAB ──────────────────────────────────────────────────────
    with tab_season:

        st.markdown('<div class="section-header">Season Overview</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(stat_box("Team", player_row['team']), unsafe_allow_html=True)
        with col2:
            st.markdown(stat_box("Fluidity Score", f"{player_row['season_fluidity']:.1f}%"), unsafe_allow_html=True)
        with col3:
            st.markdown(stat_box("All Leagues Rank", f"#{int(player_row['all_leagues_f_rank'])}"), unsafe_allow_html=True)
        with col4:
            st.markdown(stat_box("League Rank", f"#{int(player_row['league_f_rank'])}"), unsafe_allow_html=True)

        st.markdown("---")

        col_map, col_swarm = st.columns([1, 1])

        with col_map:
            st.markdown('<div class="section-header">Touch Locations</div>', unsafe_allow_html=True)
            st.markdown("<div style='font-size:1rem; font-weight:600; color:#2c3e50; margin-bottom:6px;'>Attacking direction: left → right</div>", unsafe_allow_html=True)

            player_touches = load_touches(player_id)
            season_touches = player_touches[player_touches['position_norm'] == player_row['position_norm']]

            fig, ax = plt.subplots(figsize=(8, 6.5))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')
            draw_binned_touch_map(season_touches, ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_swarm:
            st.markdown('<div class="section-header">Position Category Distribution</div>', unsafe_allow_html=True)

            pos_data = season_player_stats[
                season_player_stats['position_cat'] == selected_position
            ].copy()

            pos_avg = pos_data['season_fluidity'].mean()

            fig, ax = plt.subplots(figsize=(6, 7))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')

            other_players = pos_data[pos_data['name'] != selected_player_name]
            if len(other_players) > 0:
                sns.swarmplot(
                    data=other_players, x='position_cat', y='season_fluidity',
                    color='#adb5bd', size=6, alpha=0.6, ax=ax
                )

            player_pos_data = pos_data[pos_data['name'] == selected_player_name]
            if len(player_pos_data) > 0:
                ax.scatter(
                    [0], player_pos_data['season_fluidity'].values[0],
                    color='#e74c3c', s=120, zorder=10,
                    edgecolors='black', linewidths=1.5
                )

            ax.axhline(y=pos_avg, color='#4a7fb5', linestyle='--',
                       linewidth=1.2, alpha=0.7,
                       label=f'Position avg: {pos_avg:.1f}%')
            ax.set_xlabel(selected_position, fontsize=10, fontweight='bold')
            ax.set_ylabel('Fluidity Score (%)', fontsize=10, fontweight='bold')
            ax.set_ylim(30,80)
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.grid(axis='y', alpha=0.2)
            ax.legend(fontsize=11, frameon=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # Similar players
        st.markdown(f'<div class="section-header">Most Similar {selected_position}s (All Leagues)</div>',
                    unsafe_allow_html=True)

        all_pos_players = season_player_stats[
            (season_player_stats['position_cat'] == selected_position) &
            (season_player_stats['name'] != selected_player_name)
        ].copy()
        all_pos_players['pct_diff'] = abs(all_pos_players['season_fluidity'] - player_row['season_fluidity'])
        similar = all_pos_players.nsmallest(3, 'pct_diff')[
            ['name', 'team', 'competition', 'season_fluidity', 'pct_diff']
        ]

        col1, col2, col3 = st.columns(3)
        for i, (col, (_, row)) in enumerate(zip([col1, col2, col3], similar.iterrows())):
            with col:
                st.markdown(f"""
                <div class='info-card' style='text-align:center;'>
                    <div style='font-size:1.1rem; font-weight:700; color:#2c3e50;'>#{i+1} {row['name']}</div>
                    <div style='color:#6c757d; font-size:0.85rem;'>{row['team']} · {row['competition']}</div>
                    <div style='font-size:1.2rem; font-weight:700; color:#4a7fb5; margin-top:6px;'>
                        {row['season_fluidity']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── MATCH STATS TAB ───────────────────────────────────────────────────────
    with tab_match:

        player_matches = player_match_stats[
            (player_match_stats['player_id'] == player_id) &
            (player_match_stats['position_norm'] == player_row['position_norm'])
        ].copy()

        if len(player_matches) == 0:
            st.info("No match data available.")
            st.stop()

        # Get opponent
        def get_opponent(match_id, team):
            same_match = team_match_stats[team_match_stats['match_id'] == match_id]
            opps = same_match[same_match['team'] != team]['team'].values
            return opps[0] if len(opps) > 0 else 'Unknown'

        ha_lookup = team_match_stats[['match_id', 'team', 'home_away']].drop_duplicates()
        player_matches = player_matches.merge(ha_lookup, on=['match_id', 'team'], how='left')
        player_matches['opponent'] = player_matches.apply(
            lambda r: get_opponent(r['match_id'], r['team']), axis=1
        )
        player_matches = player_matches.sort_values('opponent').reset_index(drop=True)

        match_options = [
            f"vs {row['opponent']} ({row['home_away']}) {row['fluidity_pct']:.1f}%"
            for _, row in player_matches.iterrows()
        ]
        match_id_map = {
            opt: row['match_id']
            for opt, (_, row) in zip(match_options, player_matches.iterrows())
        }

        st.markdown("<div style='font-size:1.1rem; font-weight:700; color:#2c3e50; margin-bottom:4px;'>Select Match</div>",
                    unsafe_allow_html=True)
        selected_match_opt = st.selectbox("Select Match", match_options,
                                           key="player_match_select",
                                           label_visibility="collapsed")
        selected_match_id = match_id_map[selected_match_opt]
        match_row_pl = player_matches[player_matches['match_id'] == selected_match_id].iloc[0]

        st.markdown("---")
        st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

        total_matches_pl = len(player_matches)
        match_rank_pl = int((player_matches['fluidity_pct'] > match_row_pl['fluidity_pct']).sum()) + 1

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(stat_box("Match Fluidity Score", f"{match_row_pl['fluidity_pct']:.1f}%"), unsafe_allow_html=True)
        with col2:
            st.markdown(stat_box("Season Rank", f"#{match_rank_pl} of {total_matches_pl} matches"), unsafe_allow_html=True)

        st.markdown("---")

        # Match touch map
        player_touches_all = load_touches(player_id)
        match_locs = player_touches_all[
            (player_touches_all['match_id'] == selected_match_id) &
            (player_touches_all['position_norm'] == player_row['position_norm'])
        ]

        n_touches = len(match_locs)
        st.markdown(f'<div class="section-header">Touch Locations: {n_touches} touches</div>',
                    unsafe_allow_html=True)
        st.markdown("<div style='font-size:1rem; font-weight:600; color:#2c3e50; margin-bottom:6px;'>Attacking direction: left → right</div>", unsafe_allow_html=True)

        col_mmap, _ = st.columns([2, 1])
        with col_mmap:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')
            draw_binned_touch_map(match_locs, ax)
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

    st.markdown('<div class="section-header">Team Selection</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        leagues = sorted(avg_team_fluidity['competition'].unique())
        selected_league = st.selectbox("League", leagues, key="team_league")

    with col2:
        league_teams_df = avg_team_fluidity[
            avg_team_fluidity['competition'] == selected_league
        ].sort_values('season_z', ascending=False)
        league_teams = [
            f"{row['team']} (z={row['season_z']:.2f})"
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

    # ── TEAM SEASON TAB ───────────────────────────────────────────────────────
    with tab_team_season:

        st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

        # Get average xg and field tilt from match stats
        team_xg_avg = team_matches['shot_statsbomb_xg'].mean()
        team_ft_avg = team_matches['field_tilt'].mean()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(stat_box("Season Fluidity Z-Score", f"{team_row['season_z']:.3f}"), unsafe_allow_html=True)
        with col2:
            st.markdown(stat_box("Matches", int(team_row['matches'])), unsafe_allow_html=True)
        with col3:
            st.markdown(stat_box("Avg Non-Penalty Expected Goals (NPXG)", f"{team_xg_avg:.2f}" if pd.notna(team_xg_avg) else "N/A"), unsafe_allow_html=True)
        with col4:
            st.markdown(stat_box("Avg Field Tilt %", f"{team_ft_avg:.1f}" if pd.notna(team_ft_avg) else "N/A"), unsafe_allow_html=True)

        st.markdown("---")

        # Zonal fluidity breakdown
        st.markdown('<div class="section-header">Zonal Fluidity Breakdown (Raw Percentages)</div>',
                    unsafe_allow_html=True)

        zones = ['Defence', 'Midfield', 'Attack']
        values = [
            team_row['avg_defensive_score'],
            team_row['avg_midfield_score'],
            team_row['avg_forward_score']
        ]
        global_avgs = [
            avg_team_fluidity['avg_defensive_score'].mean(),
            avg_team_fluidity['avg_midfield_score'].mean(),
            avg_team_fluidity['avg_forward_score'].mean(),
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')
        bars = ax.bar(zones, values, color='#4a7fb5', edgecolor='black',
                      linewidth=0.8, width=0.5, alpha=0.85, zorder=3)
        bar_width = 0.5
        for i, (bar, avg) in enumerate(zip(bars, global_avgs)):
            x_center = bar.get_x() + bar.get_width() / 2
            label = 'All Teams Average' if i == 0 else '_nolegend_'
            ax.plot([x_center - bar_width / 2, x_center + bar_width / 2],
                    [avg, avg], color='#e74c3c', linewidth=2,
                    linestyle='--', zorder=4, label=label)
        ax.set_ylabel('Average Fluidity (%)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=9, frameon=False)
        ax.grid(axis='y', alpha=0.25, zorder=0)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # Match-by-match line chart
        st.markdown('<div class="section-header">Matches Distribution (match-level z scores)</div>',
                    unsafe_allow_html=True)

        team_matches_sorted = team_matches.sort_values('match_id').reset_index(drop=True)
        team_matches_sorted['match_num'] = range(1, len(team_matches_sorted) + 1)

        season_avg_z = team_matches_sorted['team_match_z'].mean()

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')
        ax.plot(team_matches_sorted['match_num'], team_matches_sorted['team_match_z'],
                color='#4a7fb5', linewidth=1.5, marker='o', markersize=5, zorder=3)
        ax.axhline(season_avg_z, color='#e74c3c', linewidth=1.5, linestyle='--',
                   label=f' Team Season avg: {season_avg_z:.2f}', zorder=2)
        ax.axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5,
                   label='All Teams Average', zorder=1)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel('Match Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fluidity Z-Score', fontsize=11, fontweight='bold')
        ax.legend(fontsize=11, frameon=False)
        ax.grid(True, alpha=0.15, linestyle=':')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── TEAM MATCH TAB ────────────────────────────────────────────────────────
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

        st.markdown("<div style='font-size:1.1rem; font-weight:700; color:#2c3e50; margin-bottom:4px;'>Select Match</div>",
                    unsafe_allow_html=True)
        selected_match_label = st.selectbox("Select Match", list(match_opts.keys()),
                                             label_visibility="collapsed")
        selected_match_id = match_opts[selected_match_label]
        match_row = team_matches_sorted[team_matches_sorted['match_id'] == selected_match_id].iloc[0]
        opponent_name = selected_match_label.split("vs ")[1].split(" (")[0]

        st.markdown("---")
        st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

        total_matches_tm = len(team_matches_sorted)
        match_rank_tm = int((team_matches_sorted['team_match_z'] > match_row['team_match_z']).sum()) + 1

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(stat_box("Match Fluidity Score", f"{match_row['team_match_z']:.3f}"), unsafe_allow_html=True)
        with col2:
            st.markdown(stat_box("Season Fluidity Rank (All Team's Matches)", f"#{match_rank_tm} of {total_matches_tm} matches"), unsafe_allow_html=True)
        with col3:
            xg_val = match_row.get('shot_statsbomb_xg')
            st.markdown(stat_box("Non-Penalty Expected Goals (NPXG)", f"{xg_val:.2f}" if pd.notna(xg_val) else "N/A"), unsafe_allow_html=True)
        with col4:
            ft_val = match_row.get('field_tilt')
            st.markdown(stat_box("Field Tilt %", f"{ft_val:.1f}" if pd.notna(ft_val) else "N/A"), unsafe_allow_html=True)

        st.markdown("---")

        # Comparison to Season line plot
        st.markdown('<div class="section-header">Comparison to Season</div>', unsafe_allow_html=True)

        season_avg_z = team_matches_sorted['team_match_z'].mean()
        sel_match_row = team_matches_sorted[team_matches_sorted['match_id'] == selected_match_id]

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')
        ax.plot(team_matches_sorted['match_num'], team_matches_sorted['team_match_z'],
                color='#4a7fb5', linewidth=1.5, marker='o', markersize=5, zorder=3)
        ax.axhline(season_avg_z, color='#e74c3c', linewidth=1.5, linestyle='--',
                   label=f' Team Season avg: {season_avg_z:.2f}', zorder=2)
        ax.axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5,
                   label='All Teams Average', zorder=1)
        if len(sel_match_row) > 0:
            ax.scatter(sel_match_row['match_num'], sel_match_row['team_match_z'],
                       color='#e74c3c', s=120, zorder=5, edgecolors='black',
                       linewidths=1.5, label='Selected Match')
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel('Match Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fluidity Z-Score', fontsize=11, fontweight='bold')
        ax.legend(fontsize=11, frameon=False)
        ax.grid(True, alpha=0.15, linestyle=':')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")

        # Zonal breakdown
        st.markdown('<div class="section-header">Zonal Breakdown (Raw Percentages)</div>',
                    unsafe_allow_html=True)

        zone_nulls = {
            'Defensive': pd.isna(match_row.get('defensive_fluidity')),
            'Midfield':  pd.isna(match_row.get('midfield_fluidity')),
            'Attack':    pd.isna(match_row.get('forward_fluidity')),
        }
        missing_zones = [z for z, is_null in zone_nulls.items() if is_null]
        if missing_zones:
            st.warning(f"{', '.join(missing_zones)} fluidity data missing for this match.")

        raw_match = [
            match_row['defensive_fluidity'] if pd.notna(match_row.get('defensive_fluidity')) else None,
            match_row['midfield_fluidity']  if pd.notna(match_row.get('midfield_fluidity'))  else None,
            match_row['forward_fluidity']   if pd.notna(match_row.get('forward_fluidity'))   else None,
        ]
        all_seasonal = [
            team_row['avg_defensive_score'],
            team_row['avg_midfield_score'],
            team_row['avg_forward_score']
        ]
        all_zones = ['Defence', 'Midfield', 'Attack']

        valid = [(z, m, s) for z, m, s in zip(all_zones, raw_match, all_seasonal) if m is not None]
        if valid:
            zones_v, match_values, seasonal_values = zip(*valid)
        else:
            zones_v, match_values, seasonal_values = [], [], []

        x = np.arange(len(zones_v))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')
        ax.bar(x - width / 2, match_values, width, label='Match',
               color='#4a7fb5', edgecolor='black', linewidth=0.7, alpha=0.9, zorder=3)
        ax.bar(x + width / 2, seasonal_values, width, label=' Team Season Average',
               color='#adb5bd', edgecolor='black', linewidth=0.7, alpha=0.7, zorder=3)
        ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(zones_v, fontsize=11)
        ax.set_ylabel('Fluidity Score (%)', fontsize=11,fontweight='bold')
        ax.set_ylim(20,90)
        ax.legend(fontsize=11, frameon=False)
        ax.grid(axis='y', alpha=0.2, zorder=0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")


        st.markdown("---")
        st.markdown('<div class="section-header">Attacking Output</div>', unsafe_allow_html=True)

        col_xg_line, col_ft_line = st.columns(2)

        with col_xg_line:
            xg_data = team_matches_sorted.dropna(subset=['shot_statsbomb_xg']).copy()
            xg_season_avg = xg_data['shot_statsbomb_xg'].mean()
            xg_league_avg = team_match_stats[
                team_match_stats['competition'] == team_row['competition']
            ]['shot_statsbomb_xg'].mean()

            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')
            ax.plot(xg_data['match_num'], xg_data['shot_statsbomb_xg'],
                    color='#adb5bd', linewidth=1.2, marker='o', markersize=4, zorder=2)
            ax.axhline(xg_season_avg, color='#e74c3c', linewidth=1.2, linestyle='--',
                       label=f'Team Average: {xg_season_avg:.2f}', zorder=1)
            ax.axhline(xg_league_avg, color='#888888', linewidth=1.0, linestyle=':',
                       label=f'All Teams Average: {xg_league_avg:.2f}', zorder=1)
            sel_xg_row = xg_data[xg_data['match_id'] == selected_match_id]
            if len(sel_xg_row) > 0:
                ax.scatter(sel_xg_row['match_num'], sel_xg_row['shot_statsbomb_xg'],
                           color='#e74c3c', s=100, zorder=5, edgecolors='black', linewidths=1.2)
            ax.set_xlabel('Match Number', fontsize=11,fontweight='bold')
            ax.set_ylabel('NPXG', fontsize=11,fontweight='bold)
            ax.set_ylim(0, 6)
            ax.legend(fontsize=10, frameon=False)
            ax.grid(True, alpha=0.15, linestyle=':')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_ft_line:
            ft_data = team_matches_sorted.dropna(subset=['field_tilt']).copy()
            ft_season_avg = ft_data['field_tilt'].mean()

            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')
            ax.plot(ft_data['match_num'], ft_data['field_tilt'],
                    color='#adb5bd', linewidth=1.2, marker='o', markersize=4, zorder=2)
            ax.axhline(ft_season_avg, color='#e74c3c', linewidth=1.2, linestyle='--',
                       label=f'Team Average: {ft_season_avg:.1f}%', zorder=1)
            ax.axhline(50, color='#888888', linewidth=1.0, linestyle=':',
                       label='All Teams Average: 50%', zorder=1)
            sel_ft_row = ft_data[ft_data['match_id'] == selected_match_id]
            if len(sel_ft_row) > 0:
                ax.scatter(sel_ft_row['match_num'], sel_ft_row['field_tilt'],
                           color='#e74c3c', s=100, zorder=5, edgecolors='black', linewidths=1.2)
            ax.set_xlabel('Match Number', fontsize=11,fontweight='bold')
            ax.set_ylabel('Field Tilt %', fontsize=11,fontweight='bold')
            ax.set_ylim(0, 100)
            ax.legend(fontsize=10, frameon=False)
            ax.grid(True, alpha=0.15, linestyle=':')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
