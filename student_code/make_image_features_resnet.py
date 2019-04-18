from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from external.vqa.vqa import VQA
from external.googlenet import googlenet

from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np

from student_code.coco_dataset import COCODataset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def do(annotation_json_file_path, question_json_file_path, image_filename_pattern, image_dir, feature_save_path):

    cocodataset = COCODataset(image_dir=image_dir)
    coco_dataloader = DataLoader(cocodataset,batch_size=128,  shuffle=False, num_workers=10)

    resNet = models.__dict__['resnet18'](pretrained=True)
    resNet = list(resNet.children())[:-2]
    resNet = nn.Sequential(*resNet)
    resNet = resNet.cuda()

    for batch_id, batch_data in enumerate(coco_dataloader):
        print("Processing Batch number : " + str(batch_id))
        images = batch_data['image'].cuda()
        features = resNet(images).detach().cpu().numpy()
        for idx, fileName in enumerate(batch_data['name']):
            np.save(feature_save_path+fileName[0:-3]+"npy",features[idx])


if __name__ == "__main__":
    annotation_json_file_path = "/data/vlr-hw3/data/mscoco_val2014_annotations.json"
    question_json_file_path = "/data/vlr-hw3/data/OpenEnded_mscoco_val2014_questions.json"
    image_dir = "/data/vlr-hw3/data/val2014/"
    image_filename_pattern = "COCO_val2014_{}.jpg"
    feature_save_path = "/data/vlr-hw3/data/val2014_features_resnet/"
    do(
        annotation_json_file_path,
        question_json_file_path,
        image_filename_pattern,
        image_dir,
        feature_save_path
    )