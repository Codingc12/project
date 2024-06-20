import os
import pandas as pd
from sent2vec.vectorizer import Vectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import csv



#Word Embedding Object
class WordEmbedding:
    #Function: Constructor for Word Embedding Class
    #@attribute type: Word Embedding Type i.e. Sentence, Document
    #@attribute text: Text For Which Word Embeddings has to generated
    def __init__(self, type: str, text: pd.DataFrame, from_file: bool = False, file_path: str = '')->None:
        self.type = type
        if from_file == True:
            self.read_from_file(file_path)
        else:
            self.text = text
            self.word_emb = self.generate_word_embeddings()
    def sen2vec_(self,document: str)->list[np.ndarray]:
        list_ = []
        vectorizer = Vectorizer(pretrained_weights='distilbert-base-multilingual-cased')
        vectorizer.run(document.split('.'))
        return vectorizer.vectors

    def wordEmbeddings(self,dataframes: pd.DataFrame)->dict:
        curr_df = dataframes
        output_list = [self.sen2vec_(j) for j in curr_df[curr_df["topic"] == 'ACCIDENT'].Text]
        return output_list
    
    #@function generate_word_embeddings is used for generating word embeddings
    #it can generate two types of word embeddings, document and sentence
    def generate_word_embeddings(self)->list:
        if self.type == 'Document':
            model_word_embeddings = SentenceTransformer("bert-base-nli-mean-tokens")
            return np.array(model_word_embeddings.encode(self.text.Text,show_progress_bar=True))
        else:
            return self.wordEmbeddings(self.text)
    #@function read_from_file is used to read word embeddings from an existing file.
    def read_from_file(self, file_path)->None:
        if os.path.isfile(file_path):
            data = pd.read_csv(file_path)
            self.text = data.Text
            self.word_emb = data.Word_EMB
        else:
            raise FileNotFoundError(f"The given file {file_path} does not exist.")
    #@function write_file is used to write word embeddings to a file.
    def write_file(self)->pd.DataFrame:
        #data
        if 'DLA' in self.text.columns:
            dataframe = pd.DataFrame(columns=['ID','Text','Word_Embeddings','DLA'])
            for i in zip(self.text.ID,self.text.Text,self.word_emb,self.text.DLA):
                if dataframe.empty:
                    dataframe.loc[0]=list(i)
                else:
                    dataframe.loc[len(dataframe.index)]=list(i)
                
        else:
            dataframe = pd.DataFrame(columns=['ID','Text','Word_Embeddings'])
            for i in zip(self.text.ID,self.text.Text,self.word_emb):
                if dataframe.empty:
                    dataframe.loc[0]=list(i)
                    
                else:
                    dataframe.loc[len(dataframe.index)]=list(i)
        np.savetxt(f'{self.type}_word_emb.csv',self.word_emb)
        
                
            
        dataframe.to_csv('df.csv')   
        return dataframe
        

            
            
        
                    
    
    

