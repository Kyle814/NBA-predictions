import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

# predictive question: build a linear model predicting points differential using respective team attributes
# Y = a_1x_1 + a_2x_2 + ...
# what is x? should it be the difference between home and away? the squared difference?
# we're gonna have home be t1 and away be t2 and have it be (t1 - t2) for each

# prepare the dependent variables: we're gonna run it across every single one to start and remove the ones with low coefficients
# figure out a more sophisticated way to identify unimportant variables
# we will run it only over 2021-2022 season

stats_2021 = pd.read_csv("./CSVs/Team Stats/2021-avgs.csv") # stats per team for 2021
games_2021 = pd.read_csv("./CSVs/All_teams/nbawholeteamstrimmed.csv") # points differential per team for 2021

# pull out series from games_2021
away_series = games_2021.loc[:, 'Visitor']
home_series = games_2021.loc[:, 'Home']
points_diff = games_2021.loc[:, 'PTSDF']

# create dataframe of points to regress over
away_stats = stats_2021.merge(away_series, left_on='Tm', right_on='Visitor', how='right').drop('Tm', axis=1)
home_stats = stats_2021.merge(home_series, left_on='Tm', right_on='Home', how='right').drop('Tm', axis=1)
stats_diffs = home_stats.drop('Home', axis=1) - away_stats.drop('Visitor', axis=1)

# dependent variable is points_diff, independent variables are in stats_diffs
regresser = linear_model.LinearRegression()
regresser.fit(stats_diffs, points_diff)