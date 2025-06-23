import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


DATA_CACHE = {}


def read_data(csv_path):
    # print('path', csv_path)
    # root = '/'.join(csv_path.split('/')[:-1])
    df = pd.read_csv(csv_path)
    # df['filename'] = df['filename'].apply(lambda x: '/'.join([root, x]))
    return df


def shuffle_dim0(tensor_):
    idx = torch.randperm(tensor_.shape[0])
    return tensor_[idx]


def random_sample_dim0(tensor_, num):
    idx = torch.randperm(tensor_.shape[0])[:num]
    return tensor_[idx]


def path_to_img(img_path, img_channels, img_size, mode):
    if img_channels == 1:
        norm = transforms.Normalize(mean=[0.5], std=[0.5])
    elif img_channels == 3:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        raise Exception('Only 1-channel and 3-channel data are supported at present.')
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(45),
            transforms.ToTensor(),
            norm
        ])
        img = Image.open(img_path).convert('RGB')

        img = transform(img)
        return img
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            norm
        ])
        img = Image.open(img_path).convert('RGB')

        img = transform(img)
        return img



class ImageDataset(Dataset):
    def __init__(self, data_path, shot, query, img_channels, img_size, mode):
        super(ImageDataset, self).__init__()
        self.shot = shot
        self.query = query
        self.img_channels = img_channels
        self.img_size = img_size
        self.mode = mode

        df = read_data(data_path)
        self.label_imgPaths_dict = df.groupby('label').apply(lambda x: x['filename'].tolist()).to_dict()
        self.labels = list(self.label_imgPaths_dict.keys())
        # print('labels',self.labels)

    def __getitem__(self, item):
        label = self.labels[item]

        if label not in DATA_CACHE:
            img_paths = self.label_imgPaths_dict[label]
            data = torch.stack([path_to_img(path, self.img_channels, self.img_size, self.mode) for path in img_paths], dim=0)
            DATA_CACHE[label] = data  # shape: (num of one class samples, c, w, h)
        else:
            data = DATA_CACHE[label]

        support_data = data[:self.shot]
        query_data = data[-self.query:]
        return support_data, query_data

    def __len__(self):
        return len(self.labels)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

