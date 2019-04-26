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

        # self.word_parallel = ParallelCoattention(D=embedding_dim, k=k)
        # self.phrase_parallel = ParallelCoattention(D=embedding_dim, k=k)
        # self.sentence_parallel = ParallelCoattention(D=embedding_dim, k=k)

        self.lin_v_short = nn.Linear(512*196,1024)
        self.lin_s_short = nn.Linear(512*max_len, 1024)
        self.lin_ans_short = nn.Linear(2048,answer_vocab)

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

        # self.lin_w = nn.Linear(512,512)
        # self.lin_p = nn.Linear(1024,512)
        # self.lin_s = nn.Linear(1024,1024)
        
        # self.lin_h = nn.Linear(1024,answer_vocab)
        
        

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

        

        # f_w = self.word_parallel(image, word_embeddings)
        # f_p = self.phrase_parallel(image, kilogram)
        # f_s = self.sentence_parallel(image, q_s)

        # h_w = self.tanh(self.lin_w(f_w))
        # h_p = self.tanh(self.lin_p(torch.cat((f_p,h_w),dim=1)))
        # h_s = self.tanh(self.lin_s(torch.cat((f_s,h_p),dim=1)))
        
        # p = F.softmax(self.lin_h(h_s),dim=1)
        v_short = self.lin_v_short(image.contiguous().view(image.shape[0],-1))
        q_short = self.lin_s_short(q_s.contiguous().view(q_s.shape[0],-1))
        
        h = torch.cat((v_short,q_short),dim=1)
        
        p = F.softmax(self.lin_ans_short(h),dim=1)
        
        return p
        
        # sentence_representation = torch.empty(1)
        # phrase_representation = torch.empty(1)
        # f_w = torch.empty(1)
        # f_p = torch.empty(1)
        # f_s = torch.empty(1)
        
        # for sentence_idx in range(word_embeddings_reshaped.shape[0]):
        #     sentence = word_embeddings_reshaped[sentence_idx]
        #     sentence = sentence.permute(2,1,0)
        #     unigram = self.unigram(sentence)
        #     sentence=torch.cat((sentence,torch.zeros(512,1,1)), dim=2)
        #     bigram = self.bigram(sentence)
        #     sentence=torch.cat((sentence,torch.zeros(512,1,1)), dim=2)
        #     trigram = self.trigram(sentence)

        #     unigram = unigram.permute(2,1,0).squeeze()
        #     bigram = bigram.permute(2,1,0).squeeze()
        #     trigram = trigram.permute(2,1,0).squeeze()

        #     kilogram = torch.stack((unigram,bigram,trigram))

        #     unigram_norm = torch.norm(unigram, dim=1)
        #     bigram_norm = torch.norm(bigram, dim=1)
        #     trigram_norm = torch.norm(trigram, dim=1)

        #     kilogram_norm = torch.stack((unigram_norm, bigram_norm, trigram_norm))
        #     phrase_indices = torch.argmax(kilogram_norm, dim=0)
        #     phrase_indices = phrase_indices.repeat(512,1).permute(1,0).unsqueeze(0)
            
        #     q_p = torch.gather(kilogram, 0, phrase_indices).squeeze()
        #     if (len(phrase_representation.shape) == 1):
        #         phrase_representation = torch.transpose(q_p.view(1,q_p.shape[0],q_p.shape[1]),1,2)
        #     else:
        #         phrase_representation = torch.cat((phrase_representation,torch.transpose(q_p.view(1,q_p.shape[0],q_p.shape[1]),1,2)))
            
        #     q_s, (h_n,c_n) = self.lstm(q_p.view(-1,1,512))
        #     q_s = q_s.squeeze()
            
        #     if (len(sentence_representation.shape) == 1):
        #         sentence_representation = q_s.view(1,q_s.shape[0],q_s.shape[1])
        #     else:
        #         sentence_representation = torch.cat((sentence_representation,q_s.view(1,q_s.shape[0],q_s.shape[1])))
            
        #     # Word Level
        #     C_w = self.tanh(torch.mm(torch.mm(word_embeddings[sentence_idx], self.W_b_weight), image[sentence_idx]))

        #     H_v_w = self.tanh(torch.torch.mm(self.W_v_weight,image[sentence_idx]) + torch.mm(torch.mm(self.W_q_weight, word_embeddings[sentence_idx].view(word_embeddings[sentence_idx].shape[1], word_embeddings[sentence_idx].shape[0])), C_w))
        #     H_q_w = self.tanh(torch.torch.mm(self.W_q_weight,word_embeddings[sentence_idx].view(word_embeddings[sentence_idx].shape[1], word_embeddings[sentence_idx].shape[0])) + torch.mm(torch.mm(self.W_v_weight, image[sentence_idx]), torch.transpose(C_w,1,0)))

        #     a_v_w = F.softmax(torch.mm(self.w_hv, H_v_w).squeeze(), dim=0)
        #     a_q_w = F.softmax(torch.mm(self.w_hq, H_q_w).squeeze(), dim=0)

        #     # Phrase Level
        #     C_p = self.tanh(torch.mm(torch.mm(q_p, self.W_b_weight), image[sentence_idx]))

        #     H_v_p = self.tanh(torch.torch.mm(self.W_v_weight,image[sentence_idx]) + torch.mm(torch.mm(self.W_q_weight, q_p.view(q_p.shape[1], q_p.shape[0])), C_p))
        #     H_q_p = self.tanh(torch.torch.mm(self.W_q_weight,q_p.view(q_p.shape[1], q_p.shape[0])) + torch.mm(torch.mm(self.W_v_weight, image[sentence_idx]), torch.transpose(C_p,1,0)))

        #     a_v_p = F.softmax(torch.mm(self.w_hv, H_v_p).squeeze(), dim=0)
        #     a_q_p = F.softmax(torch.mm(self.w_hq, H_q_p).squeeze(), dim=0)

        #     # Sentence Level
        #     C_s = self.tanh(torch.mm(torch.mm(q_s, self.W_b_weight), image[sentence_idx]))

        #     H_v_s = self.tanh(torch.torch.mm(self.W_v_weight,image[sentence_idx]) + torch.mm(torch.mm(self.W_q_weight, q_s.view(q_s.shape[1], q_s.shape[0])), C_s))
        #     H_q_s = self.tanh(torch.torch.mm(self.W_q_weight,q_s.view(q_s.shape[1], q_s.shape[0])) + torch.mm(torch.mm(self.W_v_weight, image[sentence_idx]), torch.transpose(C_s,1,0)))

        #     a_v_s = F.softmax(torch.mm(self.w_hv, H_v_s).squeeze(), dim=0)
        #     a_q_s = F.softmax(torch.mm(self.w_hq, H_q_s).squeeze(), dim=0)

        #     v_w = torch.mm(a_v_w.view(1,-1), image[sentence_idx].view(image[sentence_idx].shape[1], image[sentence_idx].shape[0]))
        #     q_w = torch.mm(a_q_w.view(1,-1), word_embeddings[sentence_idx])
        #     if(len(f_w.shape) == 1):
        #         f_w = v_w + q_w
        #     else:
        #         f_w = torch.cat((f_w, (v_w + q_w)))

        #     v_p = torch.mm(a_v_p.view(1,-1), image[sentence_idx].view(image[sentence_idx].shape[1], image[sentence_idx].shape[0]))
        #     q_p = torch.mm(a_q_p.view(1,-1), word_embeddings[sentence_idx])
        #     if(len(f_p.shape) == 1):
        #         f_p = v_p + q_p
        #     else:
        #         f_p = torch.cat((f_p, (v_p + q_p)))

        #     v_s = torch.mm(a_v_s.view(1,-1), image[sentence_idx].view(image[sentence_idx].shape[1], image[sentence_idx].shape[0]))
        #     q_s = torch.mm(a_q_s.view(1,-1), word_embeddings[sentence_idx])
        #     if(len(f_s.shape) == 1):
        #         f_s = v_s + q_s
        #     else:
        #         f_s = torch.cat((f_s, (v_s + q_s)))

        # h_w = self.tanh(self.lin_w(f_w))
        # h_p = self.tanh(self.lin_p(torch.cat((f_p,h_w),dim=1)))
        # h_s = self.tanh(self.lin_s(torch.cat((f_s,h_p),dim=1)))

        # p = F.softmax(self.lin_h(h_s),dim=1)
        # return p

        # C_w = self.dp(self.tanh(torch.matmul(torch.matmul(word_embeddings.transpose(2,1), self.W_b_weight), image)))
        # C_p = self.dp(self.tanh(torch.matmul(torch.matmul(kilogram.transpose(2,1), self.W_b_weight), image)))
        # C_s = self.dp(self.tanh(torch.matmul(torch.matmul(q_s.transpose(2,1), self.W_b_weight), image)))

        # H_v_w = self.dp(self.tanh(torch.matmul(self.W_v_weight,image) + torch.matmul(torch.matmul(self.W_q_weight, word_embeddings), C_w)))
        # H_v_p = self.dp(self.tanh(torch.matmul(self.W_v_weight,image) + torch.matmul(torch.matmul(self.W_q_weight, kilogram), C_p)))
        # H_v_s = self.dp(self.tanh(torch.matmul(self.W_v_weight,image) + torch.matmul(torch.matmul(self.W_q_weight, q_s), C_s)))

        # H_q_w = self.dp(self.tanh(torch.matmul(self.W_q_weight, word_embeddings) + torch.matmul(torch.matmul(self.W_v_weight, image), C_w.transpose(2,1))))
        # H_q_p = self.dp(self.tanh(torch.matmul(self.W_q_weight, kilogram) + torch.matmul(torch.matmul(self.W_v_weight, image), C_p.transpose(2,1))))
        # H_q_s = self.dp(self.tanh(torch.matmul(self.W_q_weight, q_s) + torch.matmul(torch.matmul(self.W_v_weight, image), C_s.transpose(2,1))))
        

        # a_v_w = F.softmax(torch.matmul(self.w_hv, H_v_w), dim=1)
        # a_q_w = F.softmax(torch.matmul(self.w_hq, H_q_w), dim=1)

        # a_v_p = F.softmax(torch.matmul(self.w_hv, H_v_p), dim=1)
        # a_q_p = F.softmax(torch.matmul(self.w_hq, H_q_p), dim=1)

        # a_v_s = F.softmax(torch.matmul(self.w_hv, H_v_s), dim=1)
        # a_q_s = F.softmax(torch.matmul(self.w_hq, H_q_s), dim=1)

        # v_w = torch.bmm(a_v_w, image.transpose(2,1))
        # b_w = torch.bmm(a_q_w, word_embeddings.transpose(2,1))
        # f_w = v_w + b_w
        # f_w = f_w.view(f_w.shape[0], f_w.shape[2])

        # v_p = torch.bmm(a_v_p, image.transpose(2,1))
        # b_p = torch.bmm(a_q_p, word_embeddings.transpose(2,1))
        # f_p = v_p + b_p
        # f_p = f_p.view(f_p.shape[0], f_p.shape[2])

        # v_s = torch.bmm(a_v_s, image.transpose(2,1))
        # b_s = torch.bmm(a_q_s, q_s.transpose(2,1))
        # f_s = v_s + b_s
        # f_s = f_s.view(f_s.shape[0], f_s.shape[2])
