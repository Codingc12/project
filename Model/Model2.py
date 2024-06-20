import networkx
import numpy as np
import sklearn
import pandas as pd
import sklearn.cluster

class Spectral_Clustering_Model:
    def __init__(self, id ,dataframe: pd.DataFrame, n_components: int =100, 
                 n_clusters: int =2):
        self.id = id
        self.n_clusters = n_clusters
        self.components = n_components
        self.adjacency_matrix = self.build_adjacency_matrix(dataframe)
        self.degree_matrix = self.build_degree_matrix()
        self.laplacian_matrix = self.build_laplacian_matrix()
        
    #matrix_structure for cosine similarity: 
    #[[(1,1),(1,2),(1,3),(1,4),(1,5), (1,6), (1,n)],
    # [(2,1),(2,2),(2,3), (2,4),(2,5),(2,6),..(2,n)],
    # ...[(n,1),...,(n,n)]]
    #The matrix we get will be a scalar matrix
    @staticmethod
    def cosine_similarity(word_embeddings):
        cosine_similarity = lambda x,y: np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
        length = len(word_embeddings)
        similarity_score = np.zeros((length,length))
        for i in range(length):
            for j in range(i+1,length):
                similarity_score[i][j] = cosine_similarity(word_embeddings[i],word_embeddings[j])*10
        return similarity_score
                
                
            
    def build_adjacency_matrix(self,data_:pd.DataFrame)->np.ndarray:
        pca_model = sklearn.decomposition.PCA(n_components=self.components)
        data = pca_model.fit_transform(data_)
        print(data.shape)
        similarity_score = Spectral_Clustering_Model.cosine_similarity(data)
        indices = Spectral_Clustering_Model.knn(similarity_score)
        self.graph = networkx.Graph()
        print(similarity_score)
        for i in indices:
            print(i)
            self.graph.add_edge(self.id[i[0]],self.id[i[1]],weights=similarity_score[i[0],i[1]])
        adjacency_matrix = networkx.to_numpy_array(self.graph)
        return adjacency_matrix
        
    @staticmethod
    def knn(similarity_score):
        indices = np.argwhere(similarity_score>7)
        return indices
       
    
    def build_degree_matrix(self):
        
        degree_matrix = np.zeros((len(self.adjacency_matrix),)*2)
        filter_anonymous = lambda x: x>0
        
        for i in range(len(self.adjacency_matrix)):
            degree_matrix[i,i] = len(list(filter(filter_anonymous,self.adjacency_matrix[i])))
        return degree_matrix
            
    def build_laplacian_matrix(self):
        return np.subtract(self.degree_matrix,self.adjacency_matrix)
    def fit_model(self):
        eigen_v = np.linalg.eig(self.laplacian_matrix)
        eigen_values = eigen_v[0]
        eigen_vectors = eigen_v[1]
        fiedler_index = 0
        fiedler_value = None
        min_eigen_value = min(eigen_values)
        for i in range(len(eigen_values)):
            curr_value = eigen_values[i]
            if fiedler_value == None:
                fiedler_index = i
                fiedler_value = curr_value
            elif fiedler_value>curr_value and curr_value != min_eigen_value:
                fiedler_index = i
                fiedler_value = curr_value
            else:
                continue
        fiedler_vectors = np.transpose(eigen_vectors)[fiedler_index]
        self.model=sklearn.cluster.KMeans(n_clusters = self.n_clusters)
        self.model.fit(fiedler_vectors)
        self.labels = self.model.labels_
        self.fiedler_vectors = fiedler_vectors
        self.fiedler_value = fiedler_value
        
        