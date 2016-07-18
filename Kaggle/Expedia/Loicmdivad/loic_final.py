# coding: utf-8

"""
	author: Loïc M. DIVAD
	date: 2016-05-12
	see also: https://www.kaggle.com/c/expedia-hotel-recommendations
"""

# imports
import sys
import time
import math
import random
import pandas as pd
import numpy  as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

# contants
SEED_STATE = 1
KFOLD_NUMB = 10
TARGET = "hotel_cluster"
LOG_FORMAT = '%(asctime)s [ %(levelname)s ] : %(message)s'
NON_FEATURES = [
    "user_id", 
    "posa_continent",
    "srch_destination_id",
    "orig_destination_distance",
    "is_mobile",
    "user_location_region",
    "user_location_city"
] # + ["is_booking"]

# logging
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(format=LOG_FORMAT)
log = logging.getLogger()
log.setLevel(logging.INFO)

# util
def parseTime(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

# TODO: Delete this
def buildTheTop(df):
    top = {}
    clusters = df["kmeans"].unique()
    for c in clusters:
        c_df = df[(df["kmeans"] == c) & (df["is_booking"] == 1)]
        top["cluster_%s"%c] = c_df.hotel_cluster.value_counts().index

    top = pd.DataFrame( dict([ (k,pd.Series(v)) for k,v in top.iteritems() ]))
    top.fillna(0, inplace=True)
    return top

def complet(x):
    n = 0
    while len(x) < 5:
        hc = GLOBAL_RANK_HOTEL[n]
        n = n + 1
        if hc not in x:
            x.append(hc)
    return x

def reformat(x):
    pattern = "%s "*5
    return pattern%tuple(x[:5])

# TODO: Delete this
def expediaPredict(df, top, n=5):
    submition = "%s "*n
    df["list_hotel_cluster"] = df["kmeans"].apply(lambda y: top["cluster_%s"%y][:n].astype(int).tolist())
    df["str_hotel_cluster"]  = df["list_hotel_cluster"].apply(lambda y: submition%tuple(y))
    return df

def dataClean(df, pcaDf):

    df["date_time"] = pd.to_datetime(df["date_time"])
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format="%Y-%m-%d", errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], format="%Y-%m-%d", errors="coerce")

    props = {}
    for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
        props[prop] = getattr(df["date_time"].dt, prop)
    
    features = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    
    for prop in features:
        props[prop] = df[prop]
    
    date_props = ["month", "day", "dayofweek", "quarter"]
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')
        
    frame = pd.DataFrame(props)
    
    frame = frame.join(pcaDf, on="srch_destination_id", how='left', rsuffix="dest")
    frame = frame.drop("srch_destination_iddest", axis=1)    
    
    #frame = frame.drop(toDrop, axis=1)
    orig_dest_dist_median = frame["orig_destination_distance"].median()
    frame["orig_destination_distance"].fillna(orig_dest_dist_median, inplace=True)
    
    return frame

def scoring(df):
    topagg = {}
    groups = df.groupby(["kmeans", "hotel_cluster"])

    # building score dict
    for name, group in groups:
        dest_id = name[0]
        cluster_id = name[1]

        clicks = len(group.is_booking[group.is_booking == False])
        bookings = len(group.is_booking[group.is_booking == True])

        score = bookings + .15 * clicks

        if dest_id not in topagg.keys():
            topagg[dest_id] = {}

        topagg[dest_id][cluster_id] = score

    # building the score dataframe
    df_top_col=[]
    for dest_id in sorted(topagg.keys()):
        df_top_col.append([i[0] for i in sorted(topagg[dest_id].items(), key=operator.itemgetter(1), reverse=True)])

    return pd.DataFrame({"kmeans": sorted(topagg.keys()), "list_hotel_cluster":df_top_col})


if __name__ == "__main__":

    start_time = time.time()

    log.info("Loading the dataset : train.csv")
    DATA_TRAIN = pd.read_csv("../data/train.csv")

    log.info("Loading the dataset : destinations.csv")
    DATA_DEST = pd.read_csv("../data/destinations.csv")

    log.info("Loading the dataset : test.csv")
    DATA_TEST = pd.read_csv("../data/test.csv")

    log.info("PCA ALGORITHM RUNNNING !!! ")
    pca = PCA(n_components=8)
    dest_pca = pca.fit_transform(DATA_DEST[["d{0}".format(i + 1) for i in range(149)]])
    dest_pca = pd.DataFrame(dest_pca)
    dest_pca["srch_destination_id"] = DATA_DEST["srch_destination_id"] #index=str
    dest_pca.rename(columns={0:"d0",1:"d1",2:"d2",3:"d3",4:"d4",5:"d5",6:"d6",7:"d7"}, inplace=True)

    #selected = DATA_TEST["user_id"].unique()
    #DATA_TARGET = DATA_TRAIN[DATA_TRAIN.user_id.isin(selected)]
    #DATA_TRAIN, DUMP_TRAIN =  train_test_split(DATA_TARGET, test_size=0.60, random_state=SEED_STATE)

    log.info("Data Cleaning : data.csv and test.csv")
    train = dataClean(DATA_TRAIN, dest_pca)
    test = dataClean(DATA_TEST, dest_pca)

    train.drop(NON_FEATURES+["is_booking"], axis=1, inplace=True)
    test.drop(NON_FEATURES, axis=1, inplace=True)

    train.dropna(inplace=True)
    test.dropna(inplace=True)
    
    #train.fillna(-1, inplace=True)
    #test.fillna(-1, inplace=True)

    log.info("KMEANS ALGORITHM RUNNNING !!! ")
    kmeans = KMeans(n_clusters=100, random_state=SEED_STATE, n_jobs=-1)
    kmeans.fit(train.drop(["hotel_cluster"], axis=1), train["hotel_cluster"])

    log.info("Building User Cluster prediction")
    #train["kmeans"] = kmeans.predict(train.drop(["user_id", "is_booking", "hotel_cluster"], axis=1))
    test["kmeans"]  = kmeans.predict(test.drop(["id"], axis=1))

    log.info("Scroring in process ... ")
    scoreddf = scoring(test)

    log.info("Complete and reformat the output")
    scoreddf["list_hotel_cluster"] = scoreddf["list_hotel_cluster"].apply(complet)
    scoreddf["str_hotel_cluster"]  = scoreddf["list_hotel_cluster"].apply(reformat)

    log.info("Join the prediction with the input")
    test = test.join(scoreddf, on="kmeans", how="left", rsuffix="_")
    #result["str_hotel_cluster"].fillna(5*"%s "%tuple(GLOBAL_RANK_HOTEL[:5]), inplace=True)

    #log.info("Grouping the top prediction per user cluster")
    #theTop = buildTheTop(train)

    #log.info("Final prediction")
    #result = expediaPredict(test, theTop)

    log.info("Wrinting result ... ")
    #result["id"] = range(len(result))
    test.to_csv("../data/loic_final_prediction.csv", columns=["id", "str_hotel_cluster"], header=["id", "hotel_cluster"], index=False)

    log.info("~ • ~ END OF THE SCRIPT, DURATION: %s Hours ~ • ~"%(parseTime(time.time() - start_time)))

