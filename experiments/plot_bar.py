""" This file plots different kinds of bars presented in the paper """
import os
import sys
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font',family='Times New Roman') 
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams.update({"font.size":18})

if __name__ == "__main__":
    x = [1,2,3,4] 
    w = 0.4
    # y1 = [42.42,45.54,49.00,57.98]
    # y2 = [56.06,61.76,65.04,76.38] 
    # x_label=['ES','ES-Pol','SAFA','SAFA-Pol'] # compare aggregation and polar transform

    # y1 = [34.78, 46.18, 49.00, 57.98]
    # y2 = [46.58, 60.18, 65.04, 76.38] 
    # x_label=['SAFA*','SAFA-Pol*','SAFA','SAFA-Pol'] # compare optimizer

    # y1 = [33.20, 42.42, 46.18, 57.96]
    # y2 = [43.88, 56.06, 60.20, 76.38] 
    # x_label=['ES*','ES','Ours*','Ours'] # compare optimizer
    
    y1 = [50.58,50.58,52.88,60.66]
    y2 = [80.58,80.58,80.66,82.96] 
    x_label=['4','4,19','4,13,19','All'] # compare semantic type
    

    plt.bar(x, y1, width=w,color='red',label='Wall Street')
    plt.bar([i + w for i in x], y2, width=w, color='blue', label='Union Square')
    for x1,y1 in enumerate(y1):
        plt.text(x1+1, y1+1, y1,ha='center',fontsize=12)
    for x2,y2 in enumerate(y2):
        plt.text(x2+1+w,y2+1,y2,ha='center',fontsize=12)
    plt.xticks([i + w/2 for i in x], x_label)
    plt.xlabel('Semantic Category')
    plt.ylabel('Top-k% recall')
    plt.yticks(np.arange(0, 110, step=20))
    plt.legend(loc=2, ncol=2, fontsize=15)
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.savefig(os.path.join('results','chapter05','type.pdf'),bbox_inches='tight') # change output file name here
    plt.show()


    
            






