import networkx
import numpy as np
import sklearn
import pandas as pd
import sklearn.cluster

class Custom_Spectral_Clustering_Model:
    def __init__(self,dataframe: pd.DataFrame, n_components: int =100, 
                 n_clusters: int =2):
        self.adjacency_matrix = self.build_adjacency_matrix(dataframe)
        self.degree_matrix = self.build_degree_matrix()
        self.laplacian_matrix = self.build_laplacian_matrix()
        self.n_clusters = n_clusters
        self.components = n_components
    #matrix_structure for cosine similarity: 
    #[[(1,1),(1,2),(1,3),(1,4),(1,5), (1,6), (1,n)],
    # [(2,1),(2,2),(2,3), (2,4),(2,5),(2,6),..(2,n)],
    # ...[(n,1),...,(n,n)]]
    #The matrix we get will be a scalar matrix
    @staticmethod
    def cosine_similarity(word_embeddings):
        cosine_similarity = lambda x,y: np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
        length = len(word_embeddings)
        similarity_score = np.ones((length,length))
        for i in range(length):
            for j in range(i+1,length):
                if i<=j:
                    similarity_score[i][j] = 0
                similarity_score[i][j] = cosine_similarity(word_embeddings[i],word_embeddings[j])
        return similarity_score
                
                
            
    def build_adjacency_matrix(self,data:pd.DataFrame)->np.ndarray:
        pca_model = sklearn.decomposition.PCA(n_components=self.components)
        data = pca_model.fit_transform(data.Word_Embeddings)
        similarity_score = Custom_Spectral_Clustering_Model.cosine_similarity(data)
        indices = Custom_Spectral_Clustering_Model.knn(similarity_score)
        self.graph = networkx.Graph()
        for i in indices:
            self.graph.add_edge(data[i[1]],data[i[2]],weights=similarity_score[i[1],i[2]])
        adjacency_matrix = networkx.adjacency_matrix(self.graph)
        return adjacency_matrix
        
    @staticmethod
    def knn(similarity_score):
        indices = np.argwhere(similarity_score>0.7)
        return indices
       
    
    def build_degree_matrix(self):
        degree_matrix = np.zeros((len(self.adjacency_matrix),)*2)
        filter_anonymous = lambda x: x>0
        adjacency_numpy_array= self.adjacency_matrix.todense()
        for i in len(adjacency_numpy_array):
            degree_matrix[i,i] = len(list(filter(filter_anonymous,self.adjacency_numpy_array[i])))
        return degree_matrix
            
    def build_laplacian_matrix(self):
        return np.subtract(self.degree_matrix,self.adjacency_matrix.toarray())
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
        
        