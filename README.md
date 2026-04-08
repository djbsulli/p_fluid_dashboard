# Positional Fluidity Dashboard

An interactive data dashboard exploring positional fluidity in professional football, built using StatsBomb open event data from the 2015-16 season across the Premier League, La Liga, Bundesliga and Serie A.
This dashboard utilises both data and code from a more in-depth dissertation project on defining positional fluidity in football, and comparing fluidity to attacking output

## What it does

- Measures positional fluidity at player and team level using a zone-based percentage metric
- Players are scored by the percentage of their touches occurring outside their most frequently occupied pitch zone
- Team scores are derived by z-scoring player percentages within position categories before aggregating
- Includes correlation analysis between fluidity and attacking performance metrics (xG, field tilt)

## Pages

- **Home** — Provides background information on the concept of positional fluidity and this project's methodology
- **Player Statistics** — Players can be selected, and the dashboard displays their key stats and touch map visuals at both season and match-level.
- The most similar players by positional fluidity are also displayed.
- **Team Statistics** — Teams can be selected, and the dashboard displays key fluidity and attacking performance stats (fluidity score, Expected Goals
  Field Tilt), at both a seasonal and match-level. Key visuals are also displayed, for additional insights into how fluidity varies within a team across position category,
and alters between matches across a team's season.

## Data

StatsBomb Open Data — [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)

## Built with

Python · Streamlit · mplsoccer · matplotlib · pandas · scipy

## Live app Link

[pfluiddashboard.streamlit.app](https://pfluiddashboard-iszmmbkwzccnkttosn4cfa.streamlit.app/)

## Link to Dissertation Project Code Book
https://colab.research.google.com/drive/1Yrjqa3ti1wNpsCvMEtbzEJPNdS97DqP7?usp=sharing
