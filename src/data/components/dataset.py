import os
import cv2
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset

def generate_single_heatmap(image_size, keypoint, sigma=1.5):
    W, H = image_size
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    d2 = (x_grid - keypoint[0])**2 + (y_grid - keypoint[1])**2
    return np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)

def generate_heatmaps(image_size, keypoints, sigma=1.5):
    return torch.tensor(
        np.stack([generate_single_heatmap(image_size, kp, sigma) for kp in keypoints]),
        dtype=torch.float32
    )

class BaseDataset(Dataset):
    def __init__(self):
        self.img_files, self.annotations = self._parse_data()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        annotation = self.annotations[index]
        id = annotation['id']
        assert os.path.splitext(os.path.basename(img_file))[0] == id
        label = annotation['annotations']
        return img_file, label

    def _parse_data(self):
        img_folder = "./data/images"
        img_files = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder)])

        with open("./data/default.json", "r", encoding="utf-8") as file:
            annotations = json.load(file)['items']

        return img_files, annotations

class CervicalDataset(Dataset):
    def __init__(self, dataset=BaseDataset(), mode='train', transform=None):
        self.dataset = dataset
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_file, annotation = self.dataset[index]

        img = cv2.imread(img_file)
        assert img is not None, f"Image not found: {img_file}"

        label = [[] for _ in range(24)]
        for point in annotation:
            label[point['label_id']].extend(point['points'])

        min_x = min(p[0] for p in label)
        min_y = min(p[1] for p in label)
        max_x = max(p[0] for p in label)
        max_y = max(p[1] for p in label)

        img = img[int(min_y-3):int(max_y+3), int(min_x-3):int(max_x+3)]
        for p in label:
            p[0] -= min_x - 3
            p[1] -= min_y - 3

        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_eq = cv2.equalizeHist(img_gs)
        img_inv = 255 - img_gs

        img_out_1 = torch.stack([
            torch.tensor(img_gs), torch.tensor(img_eq), torch.tensor(img_inv)], dim=-1)

        W = img_out_1.shape[1]
        img_out_2 = torch.stack([
            torch.tensor(cv2.flip(img_gs, 1)),
            torch.tensor(cv2.flip(img_eq, 1)),
            torch.tensor(cv2.flip(img_inv, 1))], dim=-1)

        keypoints1 = [[int(min(32, max(x/4, 0))), int(min(64, max(y/4, 0)))] for (x, y) in label]
        keypoints2 = [[32 - kp[0], kp[1]] for kp in keypoints1]

        heatmap1 = generate_heatmaps((32, 64), keypoints1)
        heatmap2 = generate_heatmaps((32, 64), keypoints2)

        if self.transform:
            img_out_1 = self.transform(image=img_out_1.numpy())['image'] / 255.0
            img_out_2 = self.transform(image=img_out_2.numpy())['image'] / 255.0
        else:
            img_out_1 = img_out_1 / 255.0
            img_out_2 = img_out_2 / 255.0

        if self.mode == 'train':
            return (img_out_1, heatmap1), (img_out_2, heatmap2)
        else:
            if random.random() < 0.5:
                return img_out_1, heatmap1
            else:
                return img_out_2, heatmap2 

    
if __name__ == "__main__":
   pass