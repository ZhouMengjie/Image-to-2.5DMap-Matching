import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import re

if __name__ == "__main__":
    accuracy = []
    accuracy_fuse = []
    accuracy_tconv = []
    accuracy_proj = []

    # load filename.log
    f = open(os.path.join('logs','dgcnn_1.log'),'r')
    line = f.readline()
    while line:
        line_content = re.split('\t| ',line)
        if '/Phase[val]' in line_content[0]:
            accuracy.append(float(line_content[4]))
        line = f.readline()
    f.close()

    f = open(os.path.join('logs','dgcnn_fuse_1.log'),'r')
    line = f.readline()
    while line:
        line_content = re.split('\t| ',line)
        if '/Phase[val]' in line_content[0]:
            accuracy_fuse.append(float(line_content[4]))
        line = f.readline()
    f.close()

    f = open(os.path.join('logs','dgcnn_tconv_1.log'),'r')
    line = f.readline()
    while line:
        line_content = re.split('\t| ',line)
        if '/Phase[val]' in line_content[0]:
            accuracy_tconv.append(float(line_content[4]))
        line = f.readline()
    f.close()

    f = open(os.path.join('logs','dgcnn_projection.log'),'r')
    line = f.readline()
    while line:
        line_content = re.split('\t| ',line)
        if '/Phase[val]' in line_content[0]:
            accuracy_proj.append(float(line_content[4]))
        line = f.readline()
    f.close()
    
    # display
    plt.figure()
    plt.plot(accuracy,color='yellow',linewidth=2,linestyle='solid',label='baseline')
    plt.plot(accuracy_fuse,color='cyan',linewidth=1,linestyle='dashed',label='3d_2d(7 x 7)')
    plt.plot(accuracy_tconv,color='red',linewidth=1,linestyle='dashed',label='3d_2d(56 x 56)')
    plt.plot(accuracy_proj,color='green',linewidth=1,linestyle='dashed',label='2d_3d')
    # plt.plot(latent_loss,color='darkorchid',linewidth=1,linestyle='dashed',label='latent_loss')
    plt.title('top-1 recall')
    plt.xlabel('epoch')
    plt.ylabel('recall %')
    plt.legend()
    plt.savefig(os.path.join('results','fusion.png'),bbox_inches='tight')
    plt.show()


    
            






