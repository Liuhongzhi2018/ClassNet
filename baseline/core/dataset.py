import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE

import pandas as pd
import time
from tqdm import tqdm

class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))                 # all 11788 images name with jpg
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))   # image number and class 1-200
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))     # image number and 0/1
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_img = [scipy.misc.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            print("train_img len: ",len(self.train_img))
            # print("train_img: ",self.train_img)
            # time.sleep(5)
            # print("train_label: ",self.train_label)
            # time.sleep(5)

        if not self.is_train:
            self.test_img = [scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            print("test_img len: ",len(self.test_img))

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class Fish():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        train_file = pd.read_csv(os.path.join(self.root, 'training.csv'))
        test_file = pd.read_csv(os.path.join(self.root, 'test.csv'))
        anno_file = pd.read_csv(os.path.join(self.root, 'annotation.csv'))
        
        train_list = []
        for line in tqdm(train_file['FileID']):
            train_list.append(line+'.jpg')
        # print("train_list: ",train_list)
        time.sleep(5)
        train_label_list = []
        for line in tqdm(train_file['SpeciesID']):
            train_label_list.append(int(line))
        # print("train_label_list: ",train_label_list)
        time.sleep(5)
        test_list = []
        for line in tqdm(test_file['FileID']):
            test_list.append(line+'.jpg')
        # print("test_list: ",test_list)
        time.sleep(5)
        test_label_list = []
        for line in tqdm(anno_file['SpeciesID']):
            test_label_list.append(int(line))
        # print("test_label_list: ",test_label_list)
        time.sleep(5)

        train_file_list = [img for img in train_list]
        test_file_list = [img for img in test_list]
        
        if self.is_train:
            self.train_img = [scipy.misc.imread(os.path.join(self.root, 'data', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = train_label_list
            print("train_img len: ",len(self.train_img))
        if not self.is_train:
            self.test_img = [scipy.misc.imread(os.path.join(self.root, 'data', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label_list
            print("test_img len: ",len(self.test_img))

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


if __name__ == '__main__':
    dataset = CUB(root='./CUB_200_2011')
    print(len(dataset.train_img))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
    dataset = CUB(root='./CUB_200_2011', is_train=False)
    print(len(dataset.test_img))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])
