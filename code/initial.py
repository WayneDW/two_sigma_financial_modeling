'''
The author was the1owl, I added the comment to make his idea clear.
'''

import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

env = kagglegym.make()
data = env.reset() # configurate environment
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME] # return ['id', 'sample', 'y', 'timestamp']
col = [c for c in data.train.columns if c not in excl] # feature names: derived 0-4, fundamental 0-63, technical 0-44


train = pd.read_hdf('../input/train.h5')
train = train[col]
d_mean= train.median(axis=0) # get the median for every column

train = data.train[col]
n = train.isnull().sum(axis=1) # get the number of nulls for every row

for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c]) # check if a column is empty, if so return true. Add feature to training data
    d_mean[c + '_nan_'] = 0
train = train.fillna(d_mean) # handle missing values
train['znull'] = n # add another feature to indicate the number of missing values
n = []

rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
model1 = rfr.fit(train, data.train['y']) # train model 1

# https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
# predict based on one-index factor model
low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (data.train.y > high_y_cut)
y_is_below_cut = (data.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut) # for each row, return true or false

model2 = LinearRegression(n_jobs=-1)
model2.fit(np.array(data.train[col].fillna(d_mean).loc[y_is_within_cut, 'technical_20'].values).reshape(-1,1), data.train.loc[y_is_within_cut, 'y'])
train = []

# https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
ymean_dict = dict(data.train.groupby(["id"])["y"].median())

while True:
    test = data.features[col]
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n
    pred = data.target
    test2 = np.array(data.features[col].fillna(d_mean)['technical_20'].values).reshape(-1,1)
    # weighted average of two model results
    pred['y'] = (model1.predict(test).clip(low_y_cut, high_y_cut) * 0.65) + (model2.predict(test2).clip(low_y_cut, high_y_cut) * 0.35)
    pred['y'] = pred.apply(lambda r: 0.95 * r['y'] + 0.05 * ymean_dict[r['id']] if r['id'] in ymean_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
    data, reward, done, info = env.step(pred)
    if done:
        print("el fin ...", info["public_score"])
        break
    if data.features.timestamp[0] % 100 == 0:
        print(reward)