import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
# imports for PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# predictive question: build a linear model predicting points differential using respective team attributes
# Y = a_1x_1 + a_2x_2 + ...
# we're gonna have home be t1 and away be t2 and have the dependent variables be (t1 - t2) for each
# we will run it only over 2021-2022 season

stats_2021 = pd.read_csv("./CSVs/Team Stats/2021-avgs.csv") # stats per team for 2021
games_2021 = pd.read_csv("./CSVs/All_teams/nbawholeteamstrimmed.csv") # points differential per team for 2021

# pull out series from games_2021
away_series = games_2021.loc[:, 'Visitor']
home_series = games_2021.loc[:, 'Home']
points_diff = games_2021.loc[:, 'PTSDF']

# create dataframe of points to perform linear regression over
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
# this does less well! our fit only hits 0.08 score, so looks like it's better to use every data point
# including PTS improves the score as well, indicating overfitting wasn't the problem

# now, we're gonna try PCA
stats_diffs_scaled = StandardScaler().fit(stats_diffs).transform(stats_diffs)
pca_all = PCA(random_state=5049)
pca_all.fit(stats_diffs_scaled)
culumative_variance_sum = np.cumsum(pca_all.explained_variance_ratio_ * 100)
# results in variance_for_pca.png, tell us 95% of the variance is in only seven components
# and almost 99% in only 10 components
pca_seven = PCA(n_components=7, random_state=5049)
transformed_stats_diffs = pca_seven.fit_transform(stats_diffs_scaled)
pca_regresser = linear_model.LinearRegression()
pca_regresser.fit(transformed_stats_diffs, points_diff)
print(pca_regresser.score(transformed_stats_diffs, points_diff))
# does even worse actually

# logistic regression
points_diff_logits = points_diff / abs(points_diff) # -1 is away team win, 1 is home team win
logistic_regresser = linear_model.LogisticRegression().fit(stats_diffs_scaled, points_diff_logits)
print(logistic_regresser.score(stats_diffs_scaled, points_diff_logits))
# performs much better with a score of 0.650