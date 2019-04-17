from PIL import Image

import os

class COCODataset(Dataset):

    def __init__(self, image_dir, image_filename_pattern):
        self.image_dir = image_dir
        self.image_feature_dir = image_feature_dir
       
        root_dir = self.image_dir
        self.image_names = [item for item in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, item))]

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        img_fileName = self.image_names[idx]
        img_path = self.image_dir + '/' + img_fileName
        with open(img_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                img = self.transform(img)
        item = {'image'=image, 'name'=img_fileName}
        return item