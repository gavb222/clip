from PIL import Image
import requests
import torch
import pandas as pd
import numpy as np

from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    
])

class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_file='Train_GCC-training.tsv', transform=preprocess):
        self.data_file = data_file
        self.data = pd.read_csv(self.data_file, sep = '\t')
        print("urls loaded")
        self.transform = transform
        self.second_preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try_again = True
        while try_again:
            try:
                row = self.data.iloc[idx]
                image = Image.open(requests.get(row['url'], timeout = 0.5, stream=True).raw)
                text = row['text']
                image = self.transform(image)
                if image.size()[0] != 3:
                    idx = np.random.randint(0, len(self.data))
                    #print("wrong number of image channels")
                    continue
                image = self.second_preprocess(image)
                try_again = False
            except:
                idx = np.random.randint(0, len(self.data))
                #print("error loading image")
 
        return image, text