import os
import cv2
import json
import torch
import random
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

def generate_single_heatmap(image_size, keypoint, sigma=1.5):
    W, H = image_size
    heatmap = np.zeros((H, W), dtype=np.float32)
    x, y = keypoint
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    d2 = (x_grid - x) ** 2 + (y_grid - y) ** 2
    heatmap = np.exp(-d2 / (2 * sigma ** 2))
    return heatmap

def generate_heatmaps(image_size, keypoints, sigma=1.5):
    heatmaps = [generate_single_heatmap(image_size, kp, sigma) for kp in keypoints]
    stacked_heatmaps = torch.tensor(np.stack(heatmaps), dtype=torch.float32)
    return stacked_heatmaps


class BaseDataset(Dataset):
    def __init__(self):
        self.img_files, self.annotations = self._parse_data()
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_file = self.img_files[index]
        annotation = self.annotations[index]
        id = annotation['id']
        assert img_file.split("/")[-1].split(".")[0] == id
        label = annotation['annotations']

        return img_file, label
    
    def _parse_data(self):
        img_folder = "./data/images"
        img_files = sorted(os.listdir(img_folder))
        img_files = [os.path.join(img_folder, img_file) for img_file in img_files]

        annotation_files = "./data/default.json"
        with open(annotation_files, "r", encoding="utf-8") as file:
            annotations = json.load(file)
        annotations = annotations['items']

        return img_files, annotations


class CervicalDataset(Dataset):
    def __init__(self, dataset=BaseDataset(), mode='train', transform=None):
        self.dataset = dataset
        self.mode = mode
        self.transform = transform
    
    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, index):
        img_file, annotation = self.dataset.__getitem__(index)
        """get raw image"""
        img = cv2.imread(img_file)
        if img is None:
            raise FileNotFoundError(f"File {img_file} not found")

        """get raw label"""
        label = [[] for _ in range(24)]

        for point in annotation:
            label[point['label_id']].extend(point['points'])

        """crop image"""
        min_x, min_y, max_x, max_y = 10000, 10000, 0, 0
        for point in label:
            min_x = min(min_x, point[0])
            min_y = min(min_y, point[1])
            max_x = max(max_x, point[0])
            max_y = max(max_y, point[1])
        
        img = img[int(min_y-3):int(max_y+3), int(min_x-3):int(max_x+3)]

        if img.shape[0] == 0 or img.shape[1] == 0:
            print(min_x, min_y, max_x, max_y)
            raise ValueError(f"Image {img_file} is empty")

        for i in range(len(label)):
            label[i][0] = label[i][0] - min_x + 3
            label[i][1] = label[i][1] - min_y + 3


        """final input tensor"""
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_eq = cv2.equalizeHist(img_gs)
        img_inv = 255 - img_gs
        
        img_out_1 = torch.stack([torch.tensor(img_gs), torch.tensor(img_eq), torch.tensor(img_inv)], dim=-1)
        img_out_2 = torch.stack([torch.tensor(cv2.flip(img_gs, 1)), torch.tensor(cv2.flip(img_inv, 1)), torch.tensor(cv2.flip(img_eq, 1))], dim=-1)
        _, W, _ = img_out_1.shape

        if self.transform:
            transformed1 = self.transform(image=img_out_1.numpy(), keypoints=label)
            transformed2 = self.transform(image=img_out_2.numpy(), keypoints=[[W-point[0], point[1]] for point in label])
            img_out_1, keypoints1 = transformed1['image'] / 255.0, transformed1['keypoints']
            img_out_2, keypoints2 = transformed2['image'] / 255.0, transformed2['keypoints']
        else:
            img_out_1 = img_out_1 / 255.0
            img_out_2 = img_out_2 / 255.0
            keypoints1 = label
            keypoints2 = [[W-point[0], point[1]] for point in label]
        
        """generate heatmap"""
        keypoints1 = [[int(min(32, max(x/4, 0))), int(min(64, max(y/4, 0)))] for (x, y) in keypoints1]
        keypoints2 = [[int(min(32, max(x/4, 0))), int(min(64, max(y/4, 0)))] for (x, y) in keypoints2]
        heatmap1 = generate_heatmaps((32, 64), keypoints1, sigma=1.5)
        heatmap2 = generate_heatmaps((32, 64), keypoints2, sigma=1.5)
        

        """get item"""
        if self.mode == 'train':
            return (img_out_1, heatmap1), (img_out_2, heatmap2)
        else:
            prob = random.random()
            if prob < 0.5:
                return img_out_1, heatmap1
            else:
                return img_out_2, heatmap2  

    
if __name__ == "__main__":
   pass