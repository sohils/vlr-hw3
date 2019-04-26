import torch
import torch.nn as nn
from external.googlenet import googlenet


class SimpleBaselineNet2(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, question_dict_size, answer_dict_size, word_feature_szie=1024, image_feature=False):
        super().__init__()
        self.image_feature = image_feature
        # Visual features
        # self.leNet = googlenet.googlenet(pretrained=True, only_features=True)

        # Question features
        self.question_embedding = nn.Sequential(nn.Linear(question_dict_size, word_feature_szie))

        self.answering = nn.Sequential(nn.Linear(word_feature_szie, answer_dict_size))

        self.image_weight = nn.Parameter(torch.tensor(0.50), requires_grad=True)
        self.question_weight = nn.Parameter(torch.tensor(0.50), requires_grad=True)


    def forward(self, image, question_encoding):
        if(self.image_feature):
            image_features = image
        else:
            # N x 1024
            image_features = self.leNet(image)
        image_features = image_features.view(image_features.shape[0],image_features.shape[2])
        # N x 1024
        word_embeddings = self.question_embedding(question_encoding)
        wr = self.question_weight*word_embeddings
        im = self.image_weight*image_features
        
        combined = wr + im

        x = self.answering(combined)

        # Returning Softmax-ed output
        return x