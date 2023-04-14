# removes data from nbawholeteams to only leave specifically what analysis.py wants

import numpy as np
import pandas as pd

games_2021 = pd.read_csv("./CSVs/All_teams/nbawholeteams.csv")
# schema for modifying team game info
team_abbreviation_to_team_name = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BRK',
    'Chicago Bulls': 'CHI',
    'Charlotte Hornets': 'CHO',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHO',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS',
}

g201 = games_2021.drop(["Date", "Start (ET)", "Overtimes","Attend.","Arena","Winner","Home_team_win"], axis=1)
g201["PTSDF"] = games_2021["PTS.1"] - games_2021["PTS"]
g202 = g201.drop(["PTS", "PTS.1"], axis=1)
g202.replace(team_abbreviation_to_team_name, inplace=True)

g202.to_csv("./CSVs/All_teams/nbawholeteamstrimmed.csv")