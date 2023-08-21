import os
import sys
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font',family='Times New Roman') 
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams.update({"font.size":22})

if __name__ == "__main__":
    location_name = 'unionsquare5kU'
    model_name = 'dgcnn_polar_best_top1'

    recall_512 = np.load(os.path.join('results', location_name+'_'+model_name+'_'+'512FPS'+'.npy'))
    recall_1024 = np.load(os.path.join('results', location_name+'_'+model_name+'_'+'1024FPS'+'.npy'))
    recall_2048 = np.load(os.path.join('results', location_name+'_'+model_name+'_'+'2048FPS'+'.npy'))

    x = np.arange(1, len(recall_512)+1) / 50
    # display
    plt.figure()
    plt.plot(x,recall_512,color='r',linewidth=1,linestyle='dashed',marker='*',markersize=5,markevery=0.2,
            label='512 points - 61.58%')
    plt.plot(x,recall_1024,color='g',linewidth=1,linestyle='solid',marker='o',markersize=5,markevery=0.2,
            label='1024 points - 63.18%')
    plt.plot(x,recall_2048,color='b',linewidth=1,linestyle='dashed',marker='s',markersize=5,markevery=0.2,
            label='2048 points - 63.22%')
    # plt.title('Random Point Sampling')
    plt.xlabel('k (% of the dataset)')
    plt.xticks(np.arange(0, 2.5, step=0.5))
    plt.ylabel('Top-k% recall')
    plt.yticks(np.arange(50, 101, step=10))
    plt.legend(loc=4)
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.savefig(os.path.join('results','fps.pdf'),bbox_inches='tight')
    plt.show()


    
            






