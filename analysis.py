import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

# predictive question: build a linear model predicting points differential using respective team attributes
# Y = a_1x_1 + a_2x_2 + ...
# we're gonna have home be t1 and away be t2 and have the dependent variables be (t1 - t2) for each

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
naive_regresser = linear_model.LinearRegression()
naive_regresser.fit(stats_diffs, points_diff)
print(naive_regresser.score(stats_diffs, points_diff))
# our score is 0.123 - we may be overfitting. so, what are the unimportant stats?
# we find this out by checking the correlation coefficients for each column

starr = []
for stat in stats_diffs:
    regr = linear_model.LinearRegression()
    stat_col = stats_diffs.loc[:, stat].values.reshape(-1,1)
    regr.fit(stat_col, points_diff)
    score = regr.score(stat_col, points_diff)
    coef = regr.coef_[0]
    # print(stat + " r^2: " + str(score))
    # print("coef: " + str(coef))
    starr.append([stat, score, coef])
stat_correlations = pd.DataFrame.from_records(starr, columns=["Stat", "Score", "Coef"])
# turns out, none of them are very well correlated
# but the greatest five are AST, STL, FG, PTS, DRB
# next five being TRB, 3P, 2P, PF, and FGA
# so we'll retry the fit with the top ten, except PTS which is correlated with AST
stats_diffs_trimmed = stats_diffs.loc[:, ["AST","STL","FG","DRB","TRB","3P","2P","PF","FGA"]]
trimmed_regresser = linear_model.LinearRegression()
trimmed_regresser.fit(stats_diffs_trimmed, points_diff)
print(trimmed_regresser.score(stats_diffs_trimmed, points_diff))
# this does less well! looks like it's better to use every data point
# including PTS improves the score as well
# use your regression(s) on 2022-2023 data to predict who wins the championship