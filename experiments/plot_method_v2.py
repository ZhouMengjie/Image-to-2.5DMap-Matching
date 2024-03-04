""" This file compares the performance of single-modal and multi-modal method on the single-image based localization task """
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
    location_name = 'hudsonriver5kU' # change area here
    model_name1 = 'resnetsafa_dgcnn_asam_2to3_up_16' # change model here
    model_name2 = 'resnetsafa_dgcnn_asam_2to3_up'

    recall_base = np.load(os.path.join('results', location_name+'_'+model_name1+'_'+'none'+'.npy'))
    recall_ours = np.load(os.path.join('results', location_name+'_'+model_name2+'_'+'none'+'.npy'))

    x = np.arange(1, len(recall_base)+1) / 50
    markevery = [0, len(x)//4*1, len(x)//4*2, len(x)//4*3, len(x)-1]
    # display
    plt.figure()
    plt.plot(x,recall_base,color='red',linewidth=2,marker='*',markersize=5,markevery=markevery,
            linestyle='dashed',label='Baseline - {:.2f}%'.format(recall_base[0]))
    plt.plot(x,recall_ours,color='blue',linewidth=2,marker='o',markersize=5,markevery=markevery,
            linestyle='solid',label='Ours - {:.2f}%'.format(recall_ours[0]))
    plt.xlabel('k (% of the dataset)')
    plt.xticks(np.arange(0, 2.5, step=0.5))
    plt.ylabel('Top-k% recall')
    plt.yticks(np.arange(40, 101, step=10))
    plt.ylim(40,101)
    plt.legend(loc=4)
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.savefig(os.path.join('results','chapter06','multi',location_name+'_method.pdf'),bbox_inches='tight')
    plt.show()


    
            






