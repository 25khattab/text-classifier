#hello again
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# preprocessing data
data = pd.read_csv("BBC News Train.csv")
vec = TfidfVectorizer(sublinear_tf=True,min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english',max_features=1000)
vec.fit(data.Text.values)
X = vec.transform(data.Text.values)
word_features = vec.get_feature_names()

# use k-means
X_train,X_test=train_test_split(X,test_size=0.8)
model = KMeans(n_clusters=5, random_state=0)
model.fit(X)
common_words = model.cluster_centers_.argsort()[:, -1:-30:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' +', '.join(word_features[word] for word in centroid))


texts = ["Hooli stock price soared after a dip in PiedPiper revenue growth.",
         "Captain Tsubasa scores a magnificent goal for the Japanese team.",
         "Merryweather mercenaries are sent on another mission, as government oversight groups call for new sanctions.",
         "Beyoncé releases a new album, tops the charts in all of south-east Asia!",
         "You won't guess what the latest trend in data analysis is!"]

text_features = vec.transform(texts)
predictions = model.predict(text_features)
print(predictions)  # """
labels=model.labels_
data["labels"]=labels
print(data.idxmax())
category_id_df = data[['Category', 'labels']].sort_values('labels')
print(data.labels.value_counts())