from torch.utils.data import Dataset
from external.vqa.vqa import VQA
from external.googlenet import googlenet

from PIL import Image

import torch
import torchvision.transforms as transforms

import numpy as np

def do(annotation_json_file_path, question_json_file_path, image_filename_pattern, image_dir,feature_save_path):
    vqa_api_handle = VQA(annotation_json_file_path, question_json_file_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])
    leNet = googlenet.googlenet(pretrained=True, only_features=True)
    leNet = leNet.cuda()

    for idx in range(len(vqa_api_handle.dataset['annotations'])):
        ann = vqa_api_handle.dataset['annotations'][idx]
        img_num = ann['image_id']
        # print(img_num)
        img_fileName = image_filename_pattern.format("%012d"%img_num)
        img_path = image_dir + '/' + img_fileName
        print("Processing image :" + img_path)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = transform(img).cuda()
            img = img.unsqueeze(0)
            image_features = leNet(img).detach().cpu().numpy()
            np.save(feature_save_path+image_filename_pattern.format("%012d"%img_num)[0:-3]+"npy",image_features)
            


if __name__ == "__main__":
    annotation_json_file_path = "/data/vlr-hw3/data/mscoco_val2014_annotations.json"
    question_json_file_path = "/data/vlr-hw3/data/OpenEnded_mscoco_val2014_questions.json"
    image_dir = "/data/vlr-hw3/data/val2014/"
    image_filename_pattern = "COCO_val2014_{}.jpg"
    feature_save_path = "/data/vlr-hw3/data/val2014_features_googlenet/"
    do(
        annotation_json_file_path,
        question_json_file_path,
        image_filename_pattern,
        image_dir,
        feature_save_path
    )