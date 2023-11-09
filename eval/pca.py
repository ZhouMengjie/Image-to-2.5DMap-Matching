import os
import numpy as np
from sklearn.decomposition import PCA

if __name__ == '__main__':
    location_name = 'unionsquare5kU'
    model_name = 'resnetsafa_dgcnn_asam_2to3_up'
    pca_dim = 16

    database_embeddings = np.load(os.path.join('datasets', 'features', 'map2', location_name+'_'+model_name+'.npy'))
    query_embeddings = np.load(os.path.join('datasets', 'features', 'pano2', location_name+'_'+model_name+'.npy'))
    
    estimator = PCA(n_components=pca_dim)
    pca_database_embeddings = estimator.fit_transform(database_embeddings) 
    pca_query_embeddings = estimator.transform(query_embeddings)   
    np.save(os.path.join('datasets', 'features', 'map2', location_name+'_'+model_name+'_16.npy'), pca_database_embeddings)
    np.save(os.path.join('datasets', 'features', 'pano2', location_name+'_'+model_name+'_16.npy'), pca_query_embeddings)
    