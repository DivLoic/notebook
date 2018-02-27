import kagglegym
import numpy as np
import pandas as pd

# Customs imports
import math
from sklearn.linear_model import RidgeCV

def sign(a):
    return -1 if a < 0 else 1

def r_score(model, x, y):
    r2 = model.score(x, y)
    return sign(r2) * math.sqrt(abs(r2))
    
# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

columns = ['fundamental_11', 'technical_19', 'technical_20', 'technical_30']

train = observation.train[columns + ['y']]
means = train.mean()
train = train.fillna(means)


model = RidgeCV(cv=12, gcv_mode='svd')
model.fit(train[columns], train.y)

print("Score on the trainning DataSet {} ".format(r_score(model, train[columns], train.y)))


# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

while True:
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    
    #obsMean = observation.features[columns].mean()
    features = observation.features[columns].fillna(means)
    observation.target.y = model.predict(features[columns])
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break