from Model.Model2 import Spectral_Clustering_Model as scm
from WordEmbeddings import word_embeddings
import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import plotly.express as px


dataFrame = pd.read_csv("csv/target.csv")
index = pd.Index(dataFrame.ID)

word_embedding_model = word_embeddings.WordEmbedding('Document',dataFrame)

doc_emb = word_embedding_model.word_emb
topics = dataFrame['topic'].unique()
for i in topics:
    print(i)
    ids = list(dataFrame[dataFrame['topic'] == i].ID)
    begin = index.get_loc(ids[0])
    last = index.get_loc(ids[-1])
    embeddings = doc_emb[begin:last]
    for j in range(0,10):
        print(j)
        model = scm(ids,embeddings,j)
        model.fit_model(3)
    
    