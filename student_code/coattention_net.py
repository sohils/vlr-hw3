import torch
import torch.nn as nn
import torch.nn.functional as F
from student_code.parallel_coattention import ParallelCoattention


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, vocab_size, embedding_dim, max_len,answer_vocab=1000):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.unigram = nn.Conv1d(embedding_dim,embedding_dim,1)
        self.bigram = nn.Conv1d(embedding_dim,embedding_dim,2)
        self.trigram = nn.Conv1d(embedding_dim,embedding_dim,3)

        self.maxp = nn.MaxPool2d(kernel_size=(3,1))

        self.lstm = nn.LSTM(embedding_dim, embedding_dim)
        k = 512

        self.word_parallel = ParallelCoattention(D=embedding_dim, k=k)
        self.phrase_parallel = ParallelCoattention(D=embedding_dim, k=k)
        self.sentence_parallel = ParallelCoattention(D=embedding_dim, k=k)

        # self.lin_v_short = nn.Linear(512*196,1024)
        # self.lin_s_short = nn.Linear(512*max_len, 1024)
        # self.lin_ans_short = nn.Linear(2048,answer_vocab)

        # self.W_b_weight = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(embedding_dim,embedding_dim)))
        # # self.W_b_bias = nn.Parameter(torch.random((embedding_dim)))

        # self.W_v_weight = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(k,embedding_dim)))
        # # self.W_v_bias = nn.Parameter(torch.random((embedding_dim)))

        # self.W_q_weight = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(k,embedding_dim)))
        # # self.W_q_bias = nn.Parameter(torch.random((embedding_dim)))

        # self.w_hv = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(1,k)))
        # self.w_hq = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(1,k)))

        self.dp = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
        self.softm = nn.Softmax()

        self.lin_w = nn.Linear(512,512)
        self.lin_p = nn.Linear(1024,512)
        self.lin_s = nn.Linear(1024,1024)
        
        self.lin_h = nn.Linear(1024,answer_vocab)
        
        

    def forward(self, image, question_encoding):
        
        image = image.view(image.shape[0],image.shape[1],-1)
        # Image [bx512x196]

        word_embeddings = self.word_embeddings(question_encoding)
        word_embeddings = word_embeddings.transpose(2,1)
        # Word Embedding [bx512xT]
        
        unigram = self.tanh(self.unigram(word_embeddings))
        bigram = self.tanh(self.bigram(torch.cat((word_embeddings, torch.zeros(word_embeddings.shape[0],word_embeddings.shape[1],1).cuda()), dim=2)))
        trigram = self.tanh(self.trigram(torch.cat((word_embeddings, torch.zeros(word_embeddings.shape[0],word_embeddings.shape[1],2).cuda()), dim=2)))
        # All-Grams [bx512xT]

        kilogram = torch.cat((
            unigram.view(unigram.shape[0],unigram.shape[1],1,unigram.shape[2]),
            bigram.view(bigram.shape[0],bigram.shape[1],1,bigram.shape[2]),
            trigram.view(trigram.shape[0],trigram.shape[1],1,trigram.shape[2])
        ), dim=2)
        kilogram = self.maxp(kilogram)
        kilogram = kilogram.view(kilogram.shape[0],kilogram.shape[1],kilogram.shape[3])
        
        q_s, (h_n,c_n) = self.lstm(kilogram.permute(2,0,1))
        q_s = q_s.permute(1,2,0)

        

        f_w = self.word_parallel(image, word_embeddings)
        f_p = self.phrase_parallel(image, kilogram)
        f_s = self.sentence_parallel(image, q_s)

        h_w = self.tanh(self.lin_w(f_w))
        h_p = self.tanh(self.lin_p(torch.cat((f_p,h_w),dim=1)))
        h_s = self.tanh(self.lin_s(torch.cat((f_s,h_p),dim=1)))
        
        p = self.lin_h(h_s)
        # v_short = self.lin_v_short(image.contiguous().view(image.shape[0],-1))
        # q_short = self.lin_s_short(q_s.contiguous().view(q_s.shape[0],-1))
        
        # h = torch.cat((v_short,q_short),dim=1)
        
        # p = F.softmax(self.lin_ans_short(h),dim=1)
        
        return p
        