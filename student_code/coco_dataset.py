from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as transforms

import os

class COCODataset(Dataset):

    def __init__(self, image_dir):
        self.image_dir = image_dir
       
        root_dir = self.image_dir
        self.image_names = [item for item in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, item))]

        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        img_fileName = self.image_names[idx]
        img_path = self.image_dir + '/' + img_fileName
        with open(img_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                img = self.transform(img)
        item = {'image':img, 'name':img_fileName}
        return item

    def __len__(self):
        return len(self.image_names)