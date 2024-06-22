import networkx
import numpy as np
import sklearn
import pandas as pd
import sklearn.cluster
import sklearn.metrics

class Spectral_Clustering_Model:
    def __init__(self, id ,dataframe: pd.DataFrame, threshold: int,n_components: int =100, 
                 n_clusters: int =2):
        self.id = id
        self.n_clusters = n_clusters
        self.components = n_components
        self.adjacency_matrix = self.build_adjacency_matrix(dataframe, threshold)
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
                
                
            
    def build_adjacency_matrix(self,data_:pd.DataFrame, threshold: int)->np.ndarray:
        pca_model = sklearn.decomposition.PCA(n_components=self.components)
        data = pca_model.fit_transform(data_)
        self.data =data
        print(data.shape)
        similarity_score = Spectral_Clustering_Model.cosine_similarity(data)
        indices = Spectral_Clustering_Model.knn(similarity_score, threshold)
        self.graph = networkx.Graph()
        
        for i in indices:
            
            self.graph.add_edge(self.id[i[0]],self.id[i[1]],weights=similarity_score[i[0],i[1]])
        adjacency_matrix = np.zeros((len(data),)*2)
        index = pd.Index(self.id)
        for (u,v,d) in self.graph.edges(data=True):
            adjacency_matrix[index.get_loc(u),index.get_loc(v)] = d['weights']
            adjacency_matrix[index.get_loc(v),index.get_loc(u)] = d['weights']
        return adjacency_matrix
        
    @staticmethod
    def knn(similarity_score,threshold: int):
        indices = np.argwhere(similarity_score>threshold)
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
        min_eigen_value = min(eigen_values.real)
        min_eigen_pos = np.where(eigen_values.real == min_eigen_value)[0][0]
        for i in range(len(eigen_values)):
            curr_value = eigen_values.real[i]
            if fiedler_value == None:
                fiedler_index = i
                fiedler_value = curr_value
            elif fiedler_value>curr_value and curr_value != min_eigen_value:
                fiedler_index = i
                fiedler_value = curr_value
            else:
                continue
        fiedler_vectors = np.transpose(eigen_vectors)[fiedler_index]
        min_eigen_vectors = np.transpose(eigen_vectors)[min_eigen_pos]
        #print(eigen_vectors)
        #print(eigen_values)
        self.model=sklearn.cluster.AgglomerativeClustering(n_clusters = self.n_clusters)
        self.model.fit(list(zip(min_eigen_vectors.real,fiedler_vectors.real)))
        self.labels = self.model.labels_
        self.fiedler_vectors = fiedler_vectors.real
        self.fiedler_value = fiedler_value.real
        print("Silhouette Score",sklearn.metrics.silhouette_score(list(zip(min_eigen_vectors.real,fiedler_vectors.real)),self.labels))
        #np.savetxt("model_labels.csv",list(self.labels))
        print("SS",sklearn.metrics.silhouette_score(self.data,self.labels))