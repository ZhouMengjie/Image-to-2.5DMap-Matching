import os
from PIL import Image
import sys
import torch
import cv2
sys.path.append(os.getcwd())
import torchvision.transforms as transforms
import numpy as np

def tensor2img(x):
    t = transforms.Compose([transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                            transforms.ToPILImage()])
    return t(x)

if __name__ == "__main__":
    dataset_path = 'datasets'
    city = 'manhattan'
    panoid1 = 'PreXwwylmG23hnheZ__zGw'
    img1 = Image.open(os.path.join(dataset_path, 'jpegs_'+city+'_2019', panoid1+'.jpg'))
    img1.show()

    img2 = cv2.imread(os.path.join(dataset_path, 'jpegs_'+city+'_2019', panoid1+'.jpg'))
    # img2 = cv2.imdecode(os.path.join(dataset_path, 'jpegs_'+city+'_2019', panoid1+'.jpg'),cv2.IMREAD_COLOR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(img2)
    img2.show()


    t = [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    transform = transforms.Compose(t)

    img1 = transform(img1)
    img2 = transform(img2)

    x1 = tensor2img(img1)
    x1.show()

    x2 = tensor2img(img2)
    x2.show()


    # dataset_path = 'datasets'
    # city = 'manhattan'
    # panoid1 = 'PreXwwylmG23hnheZ__zGw'
    # panoid2 = '_-JRaSNjuU6-qanyWSS4gQ'
    # img1 = Image.open(os.path.join(dataset_path, 'jpegs_'+city+'_2019', panoid1+'.jpg'))
    # img2 = Image.open(os.path.join(dataset_path, 'jpegs_'+city+'_2019', panoid2+'.jpg'))   

    # t = [transforms.ToTensor(),
    #              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    # transform = transforms.Compose(t)
    # img1 = transform(img1)
    # img2 = transform(img2)

    # batch = [img1,img2]
    # batch = torch.stack(batch, dim=0) 
    # img = batch[0]
    # img_tensor = torch.tensor(img)
    # x = tensor2img(img_tensor)
    # x.show()

    # batch = batch.numpy()
    # batch = np.resize(batch, (1,3,224,448))
    # img = batch[0]
    # img_tensor = torch.tensor(img)
    # x = tensor2img(img_tensor)
    # x.show()
