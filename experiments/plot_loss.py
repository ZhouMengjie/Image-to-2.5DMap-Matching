import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    cross_loss = []
    map_loss = []
    img_loss = []
    total_loss = []
    latent_loss = []
    accuracy = []

    # load filename.log
    f = open(os.path.join('logs','dgcnn_regu_1.log'),'r')
    line = f.readline()
    while line:
        line_content = line.split(' ')
        if line_content[0] == 'mean_img_map_loss:':
            cross_loss.append(float(line_content[1]))
        if line_content[0] == 'mean_map_loss:':
            map_loss.append(float(line_content[1]))
        if line_content[0] == 'mean_image_loss:':
            img_loss.append(float(line_content[1]))
        if line_content[0] == 'mean_loss:':
            total_loss.append(float(line_content[1]))
        if line_content[0] == 'mean_latent_loss:':
            latent_loss.append(float(line_content[1]))
        line = f.readline()
    f.close()
    
    # display
    plt.figure()
    plt.plot(cross_loss,color='red',linewidth=1,linestyle='dashed',label='cross_loss')
    plt.plot(map_loss,color='cyan',linewidth=2,linestyle='solid',label='map_loss')
    plt.plot(img_loss,color='black',linewidth=1,linestyle='dashed',label='image_loss')
    plt.plot(total_loss,color='yellow',linewidth=2,linestyle='solid',label='total_loss')
    # plt.plot(latent_loss,color='darkorchid',linewidth=1,linestyle='dashed',label='latent_loss')
    plt.title('training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join('results','dgcnn_regu_loss_noa_nolt.png'),bbox_inches='tight')
    plt.show()


    
            






