import torch

if __name__ == '__main__':
    device = 'cpu'
    type = 'best_top1'
    weights = 'weights/minklocmultimodal_baseline_20220905_1734_latest_20220907_1950' + '_' + type + '.pth'
    checkpoint = torch.load(weights, map_location=device)   
    print(checkpoint[type])  
    print(checkpoint['epoch'])  
