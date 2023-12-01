""" This file achieves the performance comparison between single-modal and multi-modal method on the route-based localization task
    First, run culling_plot.m/mes_plot.m/bar.m in https://github.com/ZhouMengjie/you-are-here
    Then, save the relevant .mat files in the ./exps folder
"""
import os
import sys
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
plt.rc('font',family='Times New Roman') 
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams.update({"font.size":22})

if __name__ == "__main__":
    hr_dgcnn = sio.loadmat(os.path.join('results','exps','hudsonriver5k_2dsafapolar.mat'))['acc'][0,:]*100
    ws_dgcnn = sio.loadmat(os.path.join('results','exps','wallstreet5k_2dsafapolar.mat'))['acc'][0,:]*100
    us_dgcnn = sio.loadmat(os.path.join('results','exps','unionsquare5k_2dsafapolar.mat'))['acc'][0,:]*100

    hr_2d = sio.loadmat(os.path.join('results','exps','hudsonriver5k_2d.mat'))['acc'][0,:]*100
    ws_2d = sio.loadmat(os.path.join('results','exps','wallstreet5k_2d.mat'))['acc'][0,:]*100
    us_2d = sio.loadmat(os.path.join('results','exps','unionsquare5k_2d.mat'))['acc'][0,:]*100

    x = np.arange(0, 40)+1
    plt.figure()
    plt.plot(x,hr_dgcnn,color='red',linewidth=2,
            linestyle='solid',label='HR Ours - {:.2f}%'.format(hr_dgcnn[4]))
    plt.plot(x,ws_dgcnn,color='blue',linewidth=2,
            linestyle='solid',label='WS Ours - {:.2f}%'.format(ws_dgcnn[4]))
    plt.plot(x,us_dgcnn,color='green',linewidth=2,
             linestyle='solid',label='US Ours - {:.2f}%'.format(us_dgcnn[4]))
    
    plt.plot(x,hr_2d,color='red',linewidth=2,
            linestyle='dashed',label='HR ES - {:.2f}%'.format(hr_2d[4]))
    plt.plot(x,ws_2d,color='blue',linewidth=2,
            linestyle='dashed',label='WS ES - {:.2f}%'.format(ws_2d[4]))
    plt.plot(x,us_2d,color='green',linewidth=2,
        linestyle='dashed',label='US ES - {:.2f}%'.format(us_2d[4]))
    
    plt.xlabel('Route length')
    plt.xticks(np.arange(0, 41, step=5))
    plt.xlim(5,41)
    plt.ylabel('Top-1 localization%')
    plt.yticks(np.arange(0, 101, step=10))
    plt.ylim(60,101)
    plt.legend(loc=4,fontsize=18)
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.savefig(os.path.join('results','chapter04','mes.pdf'),bbox_inches='tight')
    plt.show()


    hr_100 = sio.loadmat(os.path.join('results','exps','hudsonriver5k_100%culling.mat'))['acc'][0,:]*100
    ws_100 = sio.loadmat(os.path.join('results','exps','wallstreet5k_100%culling.mat'))['acc'][0,:]*100
    us_100 = sio.loadmat(os.path.join('results','exps','unionsquare5k_100%culling.mat'))['acc'][0,:]*100

    hr_50 = sio.loadmat(os.path.join('results','exps','hudsonriver5k_50%culling.mat'))['acc'][0,:]*100
    ws_50 = sio.loadmat(os.path.join('results','exps','wallstreet5k_50%culling.mat'))['acc'][0,:]*100
    us_50 = sio.loadmat(os.path.join('results','exps','unionsquare5k_50%culling.mat'))['acc'][0,:]*100

    x = np.arange(0, 40)+1
    plt.figure()
    plt.plot(x,hr_100,color='red',linewidth=2,
            linestyle='solid',label='HR No-Culling - {:.2f}%'.format(hr_100[4]))
    plt.plot(x,ws_100,color='blue',linewidth=2,
            linestyle='solid',label='WS No-Culling - {:.2f}%'.format(ws_100[4]))
    plt.plot(x,us_100,color='green',linewidth=2,
             linestyle='solid',label='US No-Culling - {:.2f}%'.format(us_100[4]))
    
    plt.plot(x,hr_50,color='red',linewidth=2,
            linestyle='dashed',label='HR Culling - {:.2f}%'.format(hr_50[4]))
    plt.plot(x,ws_50,color='blue',linewidth=2,
            linestyle='dashed',label='WS Culling - {:.2f}%'.format(ws_50[4]))
    plt.plot(x,us_50,color='green',linewidth=2,
        linestyle='dashed',label='US Culling - {:.2f}%'.format(us_50[4]))
    
    plt.xlabel('Route length')
    plt.xticks(np.arange(0, 41, step=5))
    plt.xlim(5,41)
    plt.ylabel('Top-1 localization%')
    plt.yticks(np.arange(0, 101, step=10))
    plt.ylim(60,101)
    plt.legend(loc=4,fontsize=18)
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.savefig(os.path.join('results','chapter04','culling.pdf'),bbox_inches='tight')
    plt.show()


#     y = sio.loadmat(os.path.join('results','exps','data.mat'))['data']*100
#     y1 = y[:,0]
#     y2 = y[:,1]
#     y3 = y[:,2]
#     x_label=['5','10','15','20']
#     x = [1,2,3,4] 
#     w = 0.25
#     plt.bar(x, y1, width=w,color='red',label='ES')
#     plt.bar([i + w for i in x], y2, width=w, color='blue', label='SAFA-Pol')
#     plt.bar([i + 2*w for i in x], y3, width=w, color='green', label='Ours')
#     plt.xticks([i + w for i in x], x_label)
#     plt.xlabel('Route length')
#     plt.ylabel('Top-1 localization%')
#     plt.yticks(np.arange(0, 110, step=10))
#     plt.ylim(60,101)
#     plt.legend(loc=2,fontsize=15)
#     plt.grid(linestyle='dashed', linewidth=0.5)
#     plt.savefig(os.path.join('results','bar.pdf'),bbox_inches='tight')
#     plt.show()



