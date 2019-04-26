import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelCoattention(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self,D,k):
        super().__init__()
        self.W_b_weight = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(D,D)))
        self.W_v_weight = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(k,D)))
        self.W_q_weight = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(k,D)))

        self.w_hv = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(1,k)))
        self.w_hq = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(1,k)))

        self.tanh = nn.Tanh()
        self.dp = nn.Dropout(p=0.5)
    
    def forward(self, image, embedding):
        C = self.dp(self.tanh(torch.matmul(torch.matmul(embedding.transpose(2,1), self.W_b_weight), image)))

        H_v = self.dp(self.tanh(torch.matmul(self.W_v_weight,image) + torch.matmul(torch.matmul(self.W_q_weight, embedding), C)))
        H_q = self.dp(self.tanh(torch.matmul(self.W_q_weight, embedding) + torch.matmul(torch.matmul(self.W_v_weight, image), C.transpose(2,1))))

        a_v = F.softmax(torch.matmul(self.w_hv, H_v), dim=1)
        a_q = F.softmax(torch.matmul(self.w_hq, H_q), dim=1)

        v = torch.bmm(a_v, image.transpose(2,1))
        b = torch.bmm(a_q, embedding.transpose(2,1))
        f = v + b
        f = f.view(f.shape[0], f.shape[2])
        return f