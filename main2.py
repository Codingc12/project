from Model.Model2 import Spectral_Clustering_Model as scm
from WordEmbeddings import word_embeddings
import pandas as pd

if __name__ == '__main__':
    files = ['csv//target.csv',
             'csv/source.csv']
    
    Word_E = {}
    for i in files[1:]:
        data = pd.read_csv(i)
        Word_E[i] =word_embeddings.WordEmbedding('Document',data)
    a = Word_E['csv/source.csv'].write_file()

    data = pd.read_csv('df.csv')
    print(len(Word_E['csv/source.csv'].word_emb))
    model = scm(a.ID, Word_E['csv/source.csv'].word_emb)
    print(model.adjacency_matrix)
    