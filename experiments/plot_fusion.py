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
    location_name = 'wallstreet5kU'
    model_name1 = 'dgcnn_best_top1'
    model_name2 = 'dgcnn_2to3_best_top1'

    recall_global = np.load(os.path.join('results', location_name+'_'+model_name1+'_'+'global'+'.npy'))
    recall_local = np.load(os.path.join('results', location_name+'_'+model_name2+'_'+'local'+'.npy'))

    x = np.arange(1, len(recall_global)+1) / 50
    # display
    plt.figure()
    plt.plot(x,recall_global,color='red',linewidth=1,marker='*',markersize=5,markevery=0.2,
            linestyle='dashed',label='global fusion - 43.52%')
    plt.plot(x,recall_local,color='blue',linewidth=1,marker='o',markersize=5,markevery=0.2,
            linestyle='solid',label='local fusion - 39.08%')
    # plt.title('Union Square')
    plt.xlabel('k (% of the dataset)')
    plt.xticks(np.arange(0, 2.5, step=0.5))
    plt.ylabel('Top-k% recall')
    plt.yticks(np.arange(50, 101, step=10))
    plt.legend(loc=4)
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.savefig(os.path.join('results','ws_fusion.pdf'),bbox_inches='tight')
    plt.show()


    
            






