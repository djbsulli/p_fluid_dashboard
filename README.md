# Positional Fluidity Dashboard

An interactive data dashboard exploring positional fluidity in professional football, built using StatsBomb open event data from the 2015-16 season across the Premier League, La Liga, Bundesliga and Serie A.

This dashboard utilises code and pre-proccessed data from a dissertation project on defining and measuring positional fluidity in football, and examining its relationship with attacking output.

## What it does

- Displays created metrics capturing positional fluidity at a team and player level.
- Player fluidity percentages at a match level are calculated as the percentage of their touches occurring outside their most frequently occupied pitch zone (using a 9*9 pitch grid).
- These match scores are then seasonally aggregated for season-level statistics.
- Team scores are derived by z-scoring player percentages within position categories before aggregating to a team's match fluidity score.
- The mean of these match scores is calculated , defined as the 'Average Fluidity Per 90'
- Team statistics are displayed alongside key attacking performance indicators (Field tilt and non-penalty expected goals), so users can get insight on how a team's
positional fluidity impacts attacking output.

## Pages

- **Home** — Background on the concept of positional fluidity and this project's methodology
- **Player Statistics** — Seasonal and match-level fluidity scores, touch map visualisations, and most similar player comparisons
- **Team Statistics** — Seasonal and match-level fluidity and attacking performance stats (xG, field tilt), with position category breakdowns and match-by-match trends

## Data

The dashboard loads pre-processed datasets derived from StatsBomb open event data using the analysis pipeline in the dissertation code notebook linked below.

Raw data source: StatsBomb Open Data Repository — [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)

## Built with

Python - Streamlit - mplsoccer - matplotlib -pandas - seaborn - scipy - statsmodels

## Live app

[pfluiddashboard.streamlit.app](https://pfluiddashboard-iszmmbkwzccnkttosn4cfa.streamlit.app/)

## Dissertation project code

[View full analysis notebook on Google Colab](https://colab.research.google.com/drive/1Yrjqa3ti1wNpsCvMEtbzEJPNdS97DqP7?usp=sharing)
