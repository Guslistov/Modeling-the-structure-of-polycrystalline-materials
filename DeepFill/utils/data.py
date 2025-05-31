import os

import pandas as pd
import numpy as np
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


def is_image_file(fname):
    return fname.lower().endswith(IMG_EXTENSIONS)


class ImageDataset2(Dataset):
    def __init__(self, folder_path, 
                       img_shape, # [W, H, C]
                       random_crop=False, 
                       scan_subdirs=False, 
                       transforms=None
                       ):
        super().__init__()
        self.img_shape = img_shape
        self.random_crop = random_crop

        self.mode = 'RGB'
        if img_shape[2] == 1:
            self.mode = 'L' # convert to greyscale

        if scan_subdirs:
            self.data = self.make_dataset_from_subdirs(folder_path)
        else:
            self.data = [entry.path for entry in os.scandir(folder_path) 
                                              if is_image_file(entry.name)]

        self.transforms = T.ToTensor()
        if transforms != None:
            self.transforms = T.Compose(transforms + [self.transforms])

    def make_dataset_from_subdirs(self, folder_path):
        samples = []
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    samples.append(os.path.join(root, fname))

        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = pil_loader(self.data[index], self.mode)

        if self.random_crop:
            w, h = img.size
            if w < self.img_shape[0] or h < self.img_shape[1]:
                img = T.Resize(max(self.img_shape[:2]))(img)
            img = T.RandomCrop(self.img_shape[:2])(img)
        else:
            img = T.Resize(self.img_shape[:2])(img)

        
        img = self.transforms(img)
        img.mul_(2).sub_(1) # [0, 1] -> [-1, 1]

        return img

def ctf_loader(path):
    """Load CTF file and convert it to tensor"""
    df = pd.read_csv(path, engine='c', low_memory=False, sep='\s+', skiprows=4, names=["Euler1", "Euler2", "Euler3"])
    data = df.values.astype(np.float32)
    image_data = data.reshape(256, 256, 3)  / 360.0
    tensor = T.functional.to_tensor(image_data)
    return tensor

def ctf_loader_fast(path):
    """Optimized CTF file loader"""
    # 1. Чтение файла сразу в буфер (быстрее, чем pandas)
    with open(path, 'r') as f:
        # Пропускаем первые 4 строки заголовка
        for _ in range(4):
            next(f)
        # Читаем остальные данные напрямую в numpy
        data = np.loadtxt(f, dtype=np.float32)
    
    # 2. Оптимизированное преобразование
    image_data = data.reshape(256, 256, 3) / 360.0
    
    # 3. Быстрое преобразование в тензор
    # Транспонируем (H,W,C) -> (C,H,W) вручную, без to_tensor
    tensor = torch.from_numpy(image_data).permute(2, 0, 1).contiguous()
    
    return tensor

CTF_EXTENSIONS = ('.ctf', '.txt', '.dat')
def is_ctf_file(fname):
    return fname.lower().endswith(CTF_EXTENSIONS)

class EBSD_Dataset(Dataset):
    def __init__(self, folder_path, 
                 img_shape=[256, 256, 3],  # Default CTF shape
                 random_crop=False, 
                 scan_subdirs=False,):
        super().__init__()
        self.img_shape = img_shape
        self.random_crop = random_crop

        if scan_subdirs:
            self.data = self.make_dataset_from_subdirs(folder_path)
        else:
            self.data = [entry.path for entry in os.scandir(folder_path) 
                                              if is_ctf_file(entry.name)]

    def make_dataset_from_subdirs(self, folder_path):
        samples = []
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            for fname in fnames:
                if is_ctf_file(fname):
                    samples.append(os.path.join(root, fname))
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tensor = ctf_loader(self.data[index])

        transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5)
        ])
        tensor = transforms(tensor)

        random_scale = random.uniform(-0.5, 0.5)

        tensor.mul_(2+random_scale).sub_(1)
        
        return tensor