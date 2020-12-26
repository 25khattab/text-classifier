from ast import dump
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import pickle


class Clustring:
    def __init__(self):
        self.data = pd.read_csv("BBC News Train.csv")
        self.vec = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',
                                   ngram_range=(1, 2), stop_words='english', max_features=1000)
        self.vec.fit(self.data.Text.values)
        self.X = self.vec.transform(self.data.Text.values)

    def train_model(self):
        self.model = KMeans(n_clusters=5, random_state=0)
        self.model.fit(self.X)

    def save_model(self):
        joblib.dump(self.model, "k-means_BBC_Model.dat")

    def load_model(self, path):
        self.model = joblib.load(path)
        value = {}
        dic = {}
        labels = self.model.labels_
        self.data["labels"] = labels
        category = self.data[['Category', 'labels']].sort_values('Category')
        hello = category.groupby(['Category', 'labels']).size().reset_index()
        for index, row in hello.iterrows():
            value[row["Category"]] = 0
        for index, row in hello.iterrows():
            if row[0] > value[row["Category"]]:
                dic[row["Category"]] = row["labels"]
                value[row["Category"]] = row[0]
        self.id_to_category = {v: k for k, v in dic.items()}

    def predict(self, object):
        text_features = self.vec.transform(object)
        predictions = self.model.predict(text_features)
        for text, predicted in zip(object, predictions):
            print('"{}"'.format(text))
            print(
                "  - Predicted as: '{}'".format(self.id_to_category[predicted]))
            print("")


clust = Clustring()
clust.load_model("k-means_BBC_Model.dat")
texts = ["Hooli stock price soared after a dip in PiedPiper revenue growth.",
         "Captain Tsubasa scores a magnificent goal for the Japanese team.",
         "Merryweather mercenaries are sent on another mission, as government oversight groups call for new sanctions.",
         "Beyoncé releases a new album, tops the charts in all of south-east Asia!",
         "You won't guess what the latest trend in data analysis is!"]
clust.predict(texts)
