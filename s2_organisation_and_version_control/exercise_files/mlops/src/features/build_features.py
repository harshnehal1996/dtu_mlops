import torch
import numpy as np
import global_vars
import os
from models.helper import get_project_dir
from torch.utils.data import Dataset

class Data(Dataset):

    def __init__(self, image, label, transform_probability=0.2):
        self.images = image
        self.labels = label
        self.transform_probability = transform_probability
    
    def __len__(self):
        return len(self.labels)
    
    def rotate(self, tensor : torch.FloatTensor) -> torch.FloatTensor:
        if np.random.randint(2):
            return torch.flip(tensor, [0]).T.unsqueeze(0)
        else:
            return torch.flip(tensor, [1]).T.unsqueeze(0)

    def __getitem__(self, idx):        
        if global_vars.isTrain and self.labels[idx].item() != 6 and self.labels[idx].item() != 9\
                                            and np.random.uniform() < self.transform_probability:
            return self.rotate(self.images[idx]), self.labels[idx]
        
        return self.images[[idx]], self.labels[idx]


def mnist():
    base_path = os.path.join(get_project_dir(), 'data/processed')
    path = os.path.join(base_path, 'train.npz')
    data = np.load(path)
    images = torch.FloatTensor(data['images'].astype(np.float32))
    labels = torch.LongTensor(data['labels'])
    path = os.path.join(base_path, 'test.npz')
    data = np.load(path)
    test_images = torch.FloatTensor(data['images'].astype(np.float32))
    test_labels = torch.LongTensor(data['labels'])
    train_dataset = Data(images, labels)
    test_dataset = Data(test_images, test_labels)

    return train_dataset, test_dataset
