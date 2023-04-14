import numpy as np
import pandas as pd

# predictive question: build a linear model predicting points differential using respective team attributes
# Y = a_1x_1 + a_2x_2 + ...
# what is x? should it be the difference between home and away? the squared difference?
# we're gonna have home be t1 and away be t2 and have it be (t1 - t2) for each

# prepare the dependent variables: we're gonna run it across every single one to start and remove the ones with low coefficients
# figure out a more sophisticated way to identify unimportant variables
# we will run it only over 2021-2022 season


stats_2021 = pd.read_csv("./CSVs/Team Stats/2021-stats.csv") # stats per team for 2021
games_2021 = pd.read_csv("./CSVs/All_teams/nbawholeteams.csv")