import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import *


class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        # self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        self.entity = torch.from_numpy(np.array(dataset['entity_token']))
        self.entity_mask = torch.from_numpy(np.array(dataset['mask_ent']))

        print('TEXT: %d, Image: %d, label: %d, Event: %d' %
              (len(self.text), len(self.image), len(
                  self.label), len(self.event_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.mask[idx],
                self.entity[idx],
                self.entity_mask[idx]), self.label[idx], self.event_label[idx]

