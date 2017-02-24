'''
The author was the1owl, I added the comment to make his idea clear.
'''
import sys
import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

#w1 = float(sys.argv[1])
w1 = 0.8
w2 = 1 - w1


env = kagglegym.make()
data = env.reset() # configurate environment
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME] # return ['id', 'sample', 'y', 'timestamp']
col = [c for c in data.train.columns if c not in excl] # feature names: derived 0-4, fundamental 0-63, technical 0-44


train = pd.read_hdf('../input/train.h5', 'train')
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


feature_group0 = ['derived_0','derived_1','derived_2','derived_3','derived_4','fundamental_0','fundamental_1','fundamental_2','fundamental_3','fundamental_5','fundamental_6','fundamental_7','fundamental_8','fundamental_9','fundamental_10','fundamental_11','fundamental_12','fundamental_13','fundamental_14','fundamental_15','fundamental_16','fundamental_17','fundamental_18','fundamental_19','fundamental_20','fundamental_21','fundamental_22','fundamental_23','fundamental_24','fundamental_25','fundamental_26','fundamental_27','fundamental_28','fundamental_29','fundamental_30','fundamental_31','fundamental_32','fundamental_33','fundamental_34','fundamental_35','fundamental_36','fundamental_37','fundamental_38','fundamental_39','fundamental_40','fundamental_41','fundamental_42','fundamental_43','fundamental_44','fundamental_45','fundamental_46','fundamental_47','fundamental_48','fundamental_49','fundamental_50','fundamental_51','fundamental_52','fundamental_53','fundamental_54','fundamental_55','fundamental_56','fundamental_57','fundamental_58','fundamental_59','fundamental_60','fundamental_61','fundamental_62','fundamental_63']

feature_group1 = ['technical_0','technical_1','technical_2','technical_3','technical_5','technical_6','technical_7','technical_9','technical_10','technical_11','technical_12','technical_13','technical_14','technical_16','technical_17','technical_18','technical_19','technical_20','technical_21','technical_22','technical_24','technical_25','technical_27','technical_28','technical_29','technical_30','technical_31','technical_32','technical_33','technical_34','technical_35','technical_36','technical_37','technical_38','technical_39','technical_40','technical_41','technical_42','technical_43','technical_44']

feature_group2 = ['derived_0_nan_','derived_1_nan_','derived_2_nan_','derived_3_nan_','derived_4_nan_','fundamental_0_nan_','fundamental_1_nan_','fundamental_2_nan_','fundamental_3_nan_','fundamental_5_nan_','fundamental_6_nan_','fundamental_7_nan_','fundamental_8_nan_','fundamental_9_nan_','fundamental_10_nan_','fundamental_11_nan_','fundamental_12_nan_','fundamental_13_nan_','fundamental_14_nan_','fundamental_15_nan_','fundamental_16_nan_','fundamental_17_nan_','fundamental_18_nan_','fundamental_19_nan_','fundamental_20_nan_','fundamental_21_nan_','fundamental_22_nan_','fundamental_23_nan_']

feature_group3 = ['fundamental_24_nan_','fundamental_25_nan_','fundamental_26_nan_','fundamental_27_nan_','fundamental_28_nan_','fundamental_29_nan_','fundamental_30_nan_','fundamental_31_nan_','fundamental_32_nan_','fundamental_33_nan_','fundamental_34_nan_','fundamental_35_nan_','fundamental_36_nan_','fundamental_37_nan_','fundamental_38_nan_','fundamental_39_nan_']

feature_group4 = ['fundamental_40_nan_','fundamental_41_nan_','fundamental_42_nan_','fundamental_43_nan_','fundamental_44_nan_','fundamental_45_nan_','fundamental_46_nan_','fundamental_47_nan_','fundamental_48_nan_','fundamental_49_nan_','fundamental_50_nan_','fundamental_51_nan_','fundamental_52_nan_','fundamental_53_nan_','fundamental_54_nan_','fundamental_55_nan_','fundamental_56_nan_','fundamental_57_nan_']


feature_depth_4_0 = ['derived_0','derived_1','derived_2','derived_3','derived_4','fundamental_0','fundamental_1','fundamental_2','fundamental_3','fundamental_5','fundamental_6','fundamental_7','fundamental_8','fundamental_9','fundamental_10','fundamental_11','fundamental_12','fundamental_13','fundamental_14','fundamental_15','fundamental_16','fundamental_17','fundamental_18','fundamental_19','fundamental_20','fundamental_21','fundamental_22','fundamental_23','fundamental_24','fundamental_25','fundamental_26','fundamental_27','fundamental_28','fundamental_29','fundamental_30','fundamental_31','fundamental_32','fundamental_33','fundamental_34','fundamental_35','fundamental_36','fundamental_37','fundamental_38','fundamental_39','fundamental_40','fundamental_41','fundamental_42','fundamental_43','fundamental_44','fundamental_45','fundamental_46','fundamental_47','fundamental_48','fundamental_49','fundamental_50','fundamental_51','fundamental_52','fundamental_53','fundamental_54','fundamental_55','fundamental_56','fundamental_57','fundamental_58','fundamental_59','fundamental_60','fundamental_61','fundamental_62','fundamental_63','technical_0','technical_1','technical_2','technical_3','technical_5','technical_6','technical_7','technical_9','technical_10','technical_11','technical_12','technical_13','technical_14','technical_16','technical_17','technical_18','technical_19','technical_20','technical_21','technical_22','technical_24','technical_25','technical_27','technical_28','technical_29','technical_30','technical_31','technical_32','technical_33','technical_34','technical_35','technical_36','technical_37','technical_38','technical_39','technical_40','technical_41','technical_42','technical_43','technical_44']

feature_depth_4_1 = ['derived_0_nan_','derived_1_nan_','derived_2_nan_','derived_3_nan_','derived_4_nan_','fundamental_0_nan_','fundamental_1_nan_','fundamental_2_nan_','fundamental_3_nan_','fundamental_5_nan_','fundamental_6_nan_','fundamental_7_nan_','fundamental_8_nan_','fundamental_9_nan_','fundamental_10_nan_','fundamental_11_nan_','fundamental_12_nan_','fundamental_13_nan_','fundamental_14_nan_','fundamental_15_nan_','fundamental_16_nan_','fundamental_17_nan_','fundamental_18_nan_','fundamental_19_nan_','fundamental_20_nan_','fundamental_21_nan_','fundamental_22_nan_','fundamental_23_nan_','fundamental_24_nan_','fundamental_25_nan_','fundamental_26_nan_','fundamental_27_nan_','fundamental_28_nan_','fundamental_29_nan_','fundamental_30_nan_','fundamental_31_nan_']

top_features = feature_depth_4_0 + feature_depth_4_1


rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
model1 = rfr.fit(train[top_features], data.train['y']) # train model 1


importances = rfr.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfr.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")


print col
try:
    for f in range(300):
        print("%d. feature %d   feature name %s, importance (%f)" % (f + 1, indices[f],  train.columns[f], importances[indices[f]]))
except:
    pass

# https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
# predict based on one-index factor model
low_y_cut = -0.06
high_y_cut = 0.06
y_is_above_cut = (data.train.y > high_y_cut)
y_is_below_cut = (data.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut) # for each row, return true or false


#cols_to_use = ['technical_20']
cols_to_use = ['technical_30', 'technical_20']

model2 = Ridge()
model2.fit(np.array(data.train[col].fillna(d_mean).loc[y_is_within_cut, cols_to_use].values), data.train.loc[y_is_within_cut, 'y'])
train = []

# https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
ymean_dict = dict(data.train.groupby(["id"])["y"].median()) # the id means sample? Get the median y for every sample

while True:
    
    test = data.features[col]
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n # prepare testing data for model 1
    test = test[top_features]   
    pred = data.target
    test2 = np.array(data.features[col].fillna(d_mean)[cols_to_use].values) # prepare testing data for model 2
    # weighted average of two model results
    pred['y'] = (model1.predict(test).clip(low_y_cut, high_y_cut) * w1) + (model2.predict(test2).clip(low_y_cut, high_y_cut) * w2)
    #pred['y'] =  (model2.predict(test2).clip(low_y_cut, high_y_cut) * 1)
    pred['y'] = pred.apply(lambda r: 0.95 * r['y'] + 0.05 * ymean_dict[r['id']] if r['id'] in ymean_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
    data, reward, done, info = env.step(pred)
    if done:
        print("el fin ...", info["public_score"])
        break
    if data.features.timestamp[0] % 100 == 0:
        print(reward)

