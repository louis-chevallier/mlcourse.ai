import numpy as np
import pandas as pd
import os
from utillc import *
import io
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from time import time

from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tree_graph_to_png(tree, feature_names, png_file_to_save):
    #tree_str = export_graphviz(
    #    tree, feature_names=feature_names, filled=True, out_file=None
    #)
    plot_tree(tree)
    #graph = pydotplus.graph_from_dot_data(tree_str)
    #graph.write_png(png_file_to_save)
    

pd.set_option("display.precision", 2)
file = os.path.join("traffic_accidents.csv")
file = "/home/louis/Desktop/traffic_accidents.csv"
file = "/home/louis/Desktop/heart_disease.csv"
file = "/home/louis/dev/git/mlcourse.ai/data/telecom_churn.csv"


df = pd.read_csv(file).dropna(axis=1)

try :
    df = df.drop("crash_date", axis=1)
except :
    pass

try :
    df = df.drop(["Churn", "State"], axis=1)
    df["International plan"] = df["International plan"].map({"Yes": 1, "No": 0})
    df["Voice mail plan"] = df["Voice mail plan"].map({"Yes": 1, "No": 0})    
except :
    pass

df1 = df.copy()
EKOX(df.head())

def get_info(df) :
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    return s

for ct, c in zip(df.dtypes, df.columns) :
    if ct.name == "object" :
        EKON(c, ct.name)
        one_hot = pd.get_dummies(df[c], prefix=c)
        df1 = df1.drop(c, axis = 1)
        df1 = df1.join(one_hot)

EKOX(df.shape)    
EKOX(df1.shape)    
scaler = StandardScaler()
X = df1._get_numeric_data()
X_scaled = scaler.fit_transform(X)
EKOX(df.head())


tsne = TSNE(random_state=17)
tsne_repr = tsne.fit_transform(X_scaled)
plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1], alpha=0.5);
plt.show()

data = tsne_repr[:,0:2]
km = KMeans(n_clusters=4, random_state=1, n_init=10)
km.fit(data)
plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1], c=km.labels_, alpha=0.5);
plt.show()

y = km.labels_
clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=17)
clf_tree.fit(X_scaled, y)
EKOX(clf_tree)

tree_graph_to_png(
    tree=clf_tree,
    feature_names=["x1", "x2"],
    png_file_to_save="./tree.png")
plt.plot()
plt.show()
    
