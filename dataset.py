import torch.utils.data as data
import os
import PIL.Image as Image
import cv2
import numpy as np


class TusimpleDataset(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        # img_num = len(os.listdir(os.path.join(root, 'images')))
        img_list = os.listdir(os.path.join(root, 'LaneImages'))
        mask_list = os.listdir(os.path.join(root, 'train_binary'))
        imgs = []
        for file, mask in zip(img_list, mask_list):
            imgs.append([file, mask])

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        x_path = '.'.join([x_path.split('.')[0], 'jpg']) #x是jpg格式，label是png

        img_x = np.array(Image.open(os.path.join('data/LaneImages', x_path)).convert('RGB'))
        img_y = np.array(Image.open(os.path.join('data/train_binary', y_path)).convert('RGB'))
        if self.transform is not None:
            img_x = self.transform(img_x)

        if self.target_transform(img_y) is not None:
            img_y = self.target_transform(img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


class TestDataset(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        # img_num = len(os.listdir(os.path.join(root, 'images')))
        img_list = os.listdir(os.path.join(root, 'images'))
        imgs = []
        for i, pic in enumerate(img_list):
            imgs.append(pic)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = cv2.imread(os.path.join('test/images', x_path))

        if self.transform is not None:
            img_x = self.transform(img_x)

        return img_x

    def __len__(self):
        return len(self.imgs)
