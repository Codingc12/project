from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import pickle


class Model:
    #@function used to initialize model class.
    #@attribute n_clusters: indicates number of clusters, default 2.
    #@attribute n_components: indicates no. of components for pca, default 100.
    def __init__(self,data: pd.DataFrame,n_clusters: int=2, n_components: int=100, load_from_file: bool = False, file_name: str= '') -> None:
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.data = data
        if load_from_file == True:
            self.model = self.read_model()
        else:
            self.model = self.build_model()
        self.model = self.build_model()
        self.tsne_data = self.tsne()
        
    
    def build_model(self)->SpectralClustering:
        model = SpectralClustering(self.n_clusters)
        pca = PCA(self.n_components)
        self.pca_data = pca.fit_transform(self.data)
        model.fit(self.pca_data)
        return model
    
    def tsne(self)->np.array:
        tsne_model = TSNE(2)
        return tsne_model.fit_transform(self.pca_data)
    
    def predict(self,x: pd.DataFrame):
        return self.model.fit_predict(x)
    
    def save_model(self):
        with open("model.pkl","wb") as file:
            pickle.dump(self.model, file)
    
    def read_model(self):
        with open("model.pkl","rb") as file:
            return pickle.load(file)
        