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
    model_name1 = 'resnet_asam'
    model_name2 = 'resnetsafa_dgcnn_asam_2to3_up'

    recall_es = np.load(os.path.join('results', location_name+'_'+model_name1+'_'+'none'+'.npy'))
    recall_ours = np.load(os.path.join('results', location_name+'_'+model_name2+'_'+'none'+'.npy'))

    x = np.arange(1, len(recall_es)+1) / 50
    markevery = [0, len(x)//4*1, len(x)//4*2, len(x)//4*3, len(x)-1]
    # display
    plt.figure()
    plt.plot(x,recall_es,color='red',linewidth=2,marker='*',markersize=5,markevery=markevery,
            linestyle='dashed',label='ES - {:.2f}%'.format(recall_es[0]))
    plt.plot(x,recall_ours,color='blue',linewidth=2,marker='o',markersize=5,markevery=markevery,
            linestyle='solid',label='Ours - {:.2f}%'.format(recall_ours[0]))
    # plt.title('Union Square')
    plt.xlabel('k (% of the dataset)')
    plt.xticks(np.arange(0, 2.5, step=0.5))
    plt.ylabel('Top-k% recall')
    plt.yticks(np.arange(40, 101, step=10))
    plt.ylim(40,101)
    plt.legend(loc=4)
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.savefig(os.path.join('results',location_name+'_method.pdf'),bbox_inches='tight')
    plt.show()


    
            






