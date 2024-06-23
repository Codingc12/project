from Model.Model2 import Spectral_Clustering_Model as scm
from WordEmbeddings import word_embeddings
import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import plotly.express as px

if __name__ == '__main__':
    files = ['csv//target.csv',
             'csv/source.csv']
    
    Word_E = {}
    for i in files:
        data = pd.read_csv(i)
        Word_E[i] =word_embeddings.WordEmbedding('Document',data)
    a = Word_E['csv//target.csv'].write_file()

    for i in range(5,10):
        print("Threshold:", i)
        for j in range(2,11):
            print("No. of eigen vectors",j)
            model = scm(a.ID, Word_E['csv//target.csv'].word_emb, i)
            model.fit_model(j)
        
    
    #print(model.fiedler_value,model.fiedler_vectors)
    
    onehot = LabelEncoder()
    labels = onehot.fit_transform(np.array(a.DLA).reshape((-1,1)))
    tsne = TSNE(n_components=2)
    tsne_data = tsne.fit_transform(model.data)
    #x = px.scatter(tsne_data,color=model.labels)
    #x.show()
    
    