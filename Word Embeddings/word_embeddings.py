import os
import pandas as pd
from sent2vec.vectorizer import Vectorizer
from sentence_transformers import SentenceTransformer
import numpy as np




#Word Embedding Object
class WordEmbedding:
    #Function: Constructor for Word Embedding Class
    #@attribute type: Word Embedding Type i.e. Sentence, Document
    #@attribute text: Text For Which Word Embeddings has to generated
    def __init__(self, type: str, text: str, from_file: bool = False, file_path: str = '')->None:
        self.type = type
        if from_file == True:
            #self.word_emb
            #self.text = 
            pass
        else:
            self.text = text
            self.word_emb = self.generate_word_embeddings()
    def sen2vec_(document: str)->list[np.ndarray]:
        list_ = []
        vectorizer = Vectorizer(pretrained_weights='distilbert-base-multilingual-cased')
        vectorizer.run(document.split('.'))
        return vectorizer.vectors

    def wordEmbeddings(dataframes: dict)->dict:
        output_dict = {}
        for i in dataframes.keys():
            curr_df = dataframes[i]
            output_list = [sen2vec_(j) for j in curr_df[curr_df['topic'] == 'ACCIDENT'].Text]
            output_dict[i] = [k for j in output_list for k in j]
        return output_dict
    
    #@function generate_word_embeddings is used for generating word embeddings
    #it can generate two types of word embeddings, document and sentence
    def generate_word_embeddings(self)->list:
        if self.type == 'Document':
            model_word_embeddings = SentenceTransformer("bert-base-nli-mean-tokens")
            return model_word_embeddings.encode(self.text,show_progress_bar=True)
        else:
            return wordEmbeddings(self.text)
    #@function read_from_file is used to read word embeddings from an existing file.
    def read_from_file(self, file_path)->None:
        if os.path.isfile(file_path):
            pass
        else:
            raise FileNotFoundError(f"The given file {file_path} does not exist.")
    #@function write_file is used to write word embeddings to a file.
    def write_file(self):
        pass
            
            
        
                    
    
    

