import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp

env = kagglegym.make()
o = env.reset()


o.train = o.train[:1000]
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col = [c for c in o.train.columns if c not in excl]

train = pd.read_hdf('../input/train.h5')
train = train[col]
d_mean= train.median(axis=0)

train = o.train[col]
n = train.isnull().sum(axis=1)
for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c])
    d_mean[c + '_nan_'] = 0
train = train.fillna(d_mean)
train['znull'] = n
n = []

rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
model1 = rfr.fit(train, o.train['y'])
train = []

#https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
low_y_cut = -0.06
high_y_cut = 0.06
# y_is_above_cut = (o.train.y > high_y_cut)
# y_is_below_cut = (o.train.y < low_y_cut)
# y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
# model2 = LinearRegression(n_jobs=-1)
# model2.fit(np.array(o.train[col].fillna(d_mean).loc[y_is_within_cut, 'technical_20'].values).reshape(-1,1), o.train.loc[y_is_within_cut, 'y'])



col_tf = ['technical_20', 'technical_30', 'fundamental_11']
im = pp.Imputer(strategy='median')
o.train[col_tf] = im.fit_transform(o.train[col_tf])
sX = pp.StandardScaler()
o.train[col_tf] = sX.fit_transform(o.train[col_tf])
o.train['b'] = 1

y_min = o.train.y.min()
y_max = o.train.y.max()

idx = (o.train.y<y_max) & (o.train.y>y_min)

features = ['b']+col_tf
n = len(features)

learning_rate = 0.01
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,n])
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.zeros([n,1]))
alpha = tf.constant(2.0)
init = tf.global_variables_initializer()

y_ = tf.matmul(X, W)

#mse + alpha*l2
cost = tf.add(tf.reduce_mean(tf.square(y_ - Y)), tf.multiply(alpha, tf.reduce_mean(tf.square(W))))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(init)

print('Training the model:')
for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={X: o.train[features], Y: o.train[['y']].values})
    cost_history = np.append(cost_history,sess.run(cost,feed_dict={X: o.train[features],Y: o.train[['y']].values}))
    if epoch in [0,1,10,20,40,100,500,training_epochs-1]:
        print(epoch, cost_history[-1])



#https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
ymean_dict = dict(o.train.groupby(["id"])["y"].median())

while True:

    # tree part
    test = o.features[col]
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n
    


    # tf_lr part
    o.features[col_tf] = im.transform(o.features[col_tf])
    o.features[col_tf] = sX.transform(o.features[col_tf])
    o.features['b'] = 1
    
    o.target.y = sess.run(y_, feed_dict={X:o.features[features]})
    o.target.y = np.clip(o.target.y, y_min, y_max)
    # tf_lr end
    pred = o.target
    # pred['y'] = o.target.y
    pred['y'] = (model1.predict(test).clip(low_y_cut, high_y_cut) * 0.65) + o.target.y * 0.35
    pred['y'] = pred.apply(lambda r: 0.95 * r['y'] + 0.05 * ymean_dict[r['id']] if r['id'] in ymean_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
    o, reward, done, info = env.step(pred)
    if done:
        print("el fin ...", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)