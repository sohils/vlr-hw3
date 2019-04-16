from torch.utils.data import Dataset
from external.vqa.vqa import VQA
from external.googlenet import googlenet

from PIL import Image

import torch
import torchvision.transforms as transforms

def do(annotation_json_file_path, question_json_file_path, image_filename_pattern, image_dir,feature_save_path):
    vqa_api_handle = VQA(annotation_json_file_path, question_json_file_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])
    leNet = googlenet.googlenet(pretrained=True, only_features=True)

    for idx in range(len(self.vqa_api_handle.dataset['annotations'])):
        ann = vqa_api_handle.dataset['annotations'][idx]
        img_num = ann['image_id']
        img_fileName = image_filename_pattern.format("%012d"%img_num)
        img_path = image_dir + '/' + img_fileName
        print("Processing image :" + img_fileName)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = self.transform(img).cuda()
            image_features = leNet(img).cpu().numpy()
            np.save(feature_save_path+image_filename_pattern.format("%012d"%img_num)[0:-3]+"npy",image_features)
            


if __name__ == "__main__":
    annotation_json_file_path = "/data/vlr-hw3/data/mscoco_train2014_annotations.json"
    question_json_file_path = "/data/vlr-hw3/data/OpenEnded_mscoco_train2014_questions.json"
    image_dir = "/data/vlr-hw3/data/train2014/"
    image_filename_pattern = "COCO_train2014_{}.jpg"
    feature_save_path = "/data/vlr-hw3/data/train2014_features_googlenet/"
    do(
        annotation_json_file_path,
        question_json_file_path,
        image_dir,
        image_filename_pattern,
        feature_save_path
    )