import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import re

if __name__ == "__main__":
    lr3_loss = []
    lr4_loss = []
    lr5_loss = []
    
    lr3_recall = []
    lr4_recall = []
    lr5_recall = []

    # load filename.log
    # seqnet: 20231013_1848, 20231013_1734, 20231010_1344
    # transmixer: 20231012_1718, 20231010_1613, 20231010_1224
    f = open(os.path.join('arun_log','20231012_1718.txt'),'r')
    line = f.readline()
    while line:
        line_content = re.split('\t| ',line)
        if 'mean_loss' in line_content[0]:
            lr3_loss.append(float(line_content[1]))
        if '/Phase[val]' in line_content[0]:
            lr3_recall.append(float(line_content[4]))
        line = f.readline()
    f.close()

    f = open(os.path.join('arun_log','20231010_1613.txt'),'r')
    line = f.readline()
    while line:
        line_content = re.split('\t| ',line)
        if 'mean_loss' in line_content[0]:
            lr4_loss.append(float(line_content[1]))
        if '/Phase[val]' in line_content[0]:
            lr4_recall.append(float(line_content[4]))
        line = f.readline()
    f.close()

    f = open(os.path.join('arun_log','20231010_1224.txt'),'r')
    line = f.readline()
    while line:
        line_content = re.split('\t| ',line)
        if 'mean_loss' in line_content[0]:
            lr5_loss.append(float(line_content[1]))
        if '/Phase[val]' in line_content[0]:
            lr5_recall.append(float(line_content[4]))
        line = f.readline()
    f.close()
    
    # display
    plt.figure()
    plt.plot(lr3_recall,color='yellow',linewidth=2,linestyle='solid',label='lr = 1e-3')
    plt.plot(lr4_recall,color='cyan',linewidth=1,linestyle='dashed',label='lr = 1e-4')
    plt.plot(lr5_recall,color='red',linewidth=1,linestyle='dashed',label='lr = 1e-5')
    plt.title('top-1 recall')
    plt.xlabel('epoch')
    plt.ylabel('recall %')
    plt.legend()
    plt.savefig(os.path.join('results','transmixer_recall.png'),bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(lr3_loss,color='yellow',linewidth=2,linestyle='solid',label='lr = 1e-3')
    plt.plot(lr4_loss,color='cyan',linewidth=1,linestyle='dashed',label='lr = 1e-4')
    plt.plot(lr5_loss,color='red',linewidth=1,linestyle='dashed',label='lr = 1e-5')
    plt.title('training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join('results','transmixer_loss.png'),bbox_inches='tight')
    plt.show()


    
            






