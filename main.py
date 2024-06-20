from Model import Model as model
from WordEmbeddings import word_embeddings
import pandas as pd

#Sampling 
def sampling(data: pd.DataFrame)->pd.DataFrame:
    return data.sample(frac=0.2,replace='False')
if __name__ == '__main__':
    files = ['csv//target.csv',
             'csv/source.csv']
    Word_E = {}
    for i in files[:1]:
        data = pd.read_csv(i)
        Word_E[i] =word_embeddings.WordEmbedding('sent',data)
    a = Word_E['csv//target.csv'].write_file()
    for j in a.Word_Embeddings:
        print(len(j))
    #Model1 = model.Model(Word_E[files[1]].word_emb)
    #Model1.save_model()
    #Model2 = Model1.read_model()
    #print(Model2)
    #print(Model2.labels_) 
        