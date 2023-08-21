import os
import sys
sys.path.append(os.getcwd())
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

if __name__ == "__main__":
      location_name = 'unionsquare5kU'
      model_name = 'model_MinkLocMultimodal_20230717_1154_best_top1'
      # seed = 1
      # np.random.seed(seed)
      embeddings = sio.loadmat(os.path.join('results', location_name+'_'+model_name+'.mat'))   
      database_embeddings = embeddings['ref']
      print(database_embeddings)
      query_embeddings = embeddings['qry']
      
      # indexes = np.random.choice(len(query_embeddings), 5000, replace=False)
      # database_embeddings = database_embeddings[indexes]
      # query_embeddings = query_embeddings[indexes]

      # t-sne
      # X = np.append(database_embeddings,query_embeddings,axis=0)
      X = query_embeddings
      tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
      X_tsne = tsne.fit_transform(X)
      print('Org data dimension is {}. Embedded data dimension is {}'.format(X.shape[-1], X_tsne.shape[-1]))
      
      '''Embedding Visualization'''
      x_min, x_max = X_tsne.min(0), X_tsne.max(0)
      X_norm = (X_tsne - x_min) / (x_max - x_min)  # normalization
      plt.figure()
      for i in range(X_norm.shape[0]):
            if i >=0 and i < 5000:
                  color = 'b'
            else:
                  color = 'r'
            plt.scatter(X_norm[i,0], X_norm[i,1], s=10, c=color)
            # plt.text(X_norm[i, 0], X_norm[i, 1], str(i), color=plt.cm.Set1(0), 
            #          fontdict={'weight': 'bold', 'size': 9})
      plt.xticks([])
      plt.yticks([])
      plt.savefig(os.path.join('results',location_name+'_'+model_name+'_pano_tsne.jpeg'), dpi=500)
      # plt.show()
      plt.close()