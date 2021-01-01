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
        try:
            self.data = pd.read_csv("data/BBC News Train.csv")
            self.load_model()
        except FileNotFoundError:
            self.train_model()
            self.save_model()
            self.load_model()

    def train_model(self):
        
        self.train_vec = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',
                                         ngram_range=(1, 2), stop_words='english', max_features=1000)
        self.train_vec.fit(self.data.Text.values)
        self.X = self.train_vec.transform(self.data.Text.values)
        self.model = KMeans(n_clusters=5, random_state=0)
        self.model.fit(self.X)

    def save_model(self):
        joblib.dump(self.model, "data/k-means_BBC_Model.dat")
        pickle.dump(self.train_vec, open("data/vec.pickle", "wb"))

    def load_model(self):

        self.vec = pickle.load(open("data/vec.pickle", "rb"))
        self.model = joblib.load("data/k-means_BBC_Model.dat")
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

    def predict(self, txt):
        txt = [txt]
        text_features = self.vec.transform(txt)
        prediction = self.model.predict(text_features)
        return("Predicted as: '{}'".format(self.id_to_category[prediction[0]]))
