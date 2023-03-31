import numpy as np
import pandas as pd

year = "2019" # change to change the year you're working with

df = pd.read_csv("./" + year + "-stats.csv")

# extracting team totals, so we remove everything except the teams and the stats
trimmed_noteamcol = df.loc[:, "FG":"PTS"]
trimmed_teamcol = df.loc[:, 'Tm']
trimmed_teams = pd.merge(trimmed_noteamcol, trimmed_teamcol, left_index = True, right_index = True)

# group and average by team
team_avgs = trimmed_teams.groupby('Tm').mean()
team_avgs.to_csv("./" + year + "-avgs.csv")