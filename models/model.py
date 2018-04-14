#coding:utf8
import numpy 
import torch as t
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from .basicmodule import BasicModule
t.backends.cudnn.enabled=False
def one_hot(ids, out_tensor):
    """
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """
    if not isinstance(ids, (list, np.ndarray)):
        raise ValueError("ids must be 1-D list or array")
    ids = torch.LongTensor(ids).view(-1,1)
    out_tensor.zero_()
    out_tensor.scatter_(dim=1, index=ids, src=1.)
    # out_tensor.scatter_(1, ids, 1.0)

class EVQA(BasicModule):
        def __init__(self,data_params,model_params):
                super(EVQA,self).__init__()
                self.input_frame_dim = data_params['input_video_dim']
                self.input_n_frames = data_params['max_n_frames']
                self.input_ques_dim = data_params['input_ques_dim']
                self.max_n_q_words = data_params['max_n_q_words']
                self.max_n_a_words = data_params['max_n_a_words']
                self.n_words = data_params['n_words']

                self.ref_dim =  model_params['ref_dim']
                self.lstm_dim = model_params['lstm_dim']
                self.attention_dim = model_params['attention_dim']
                self.regularization_beta = model_params['regularization_beta']
                self.dropout_prob = model_params['dropout_prob']

                self.decode_dim = model_params['decode_dim']
                self.video_rnn = nn.LSTM(self.input_frame_dim,self.input_frame_dim,1,batch_first=False)
                self.ques_rnn = nn.LSTM(self.input_ques_dim,self.input_ques_dim,1,batch_first=False)
                self.video_linear = nn.Linear(self.input_frame_dim,self.attention_dim,bias=True)
                self.ques_linear = nn.Linear(self.input_ques_dim,self.attention_dim,bias=True)
                self.decoder_linear = nn.Linear(self.attention_dim,self.decode_dim,bias=True)
                self.embed_word_linear = nn.Linear(self.decode_dim,self.n_words,bias=True)
                self.word_to_lstm_linear = nn.Linear(self.input_ques_dim,self.decode_dim,bias=True)
                self.decoder_cell = nn.LSTMCell(self.decode_dim,self.decode_dim)

        def forward(self,input_q,input_q_len,input_x,input_x_len,y,y_mask,ans_vec,batch_size):
                video_h0 = Variable(t.zeros(1,batch_size,self.input_frame_dim)).cuda()
                video_c0 = Variable(t.zeros(1,batch_size,self.input_frame_dim)).cuda()

                ques_h0 = Variable(t.zeros(1,batch_size,self.input_ques_dim)).cuda()
                ques_c0 = Variable(t.zeros(1,batch_size,self.input_ques_dim)).cuda()

                decoder_h0 = Variable(t.zeros(1,batch_size,self.decode_dim)).cuda()
                decoder_c0 = Variable(t.zeros(1,batch_size,self.decode_dim)).cuda()

                video_output,(video_hn,video_cn) = self.video_rnn(input_x.transpose(0,1).float(),(video_h0,video_c0))
                ques_output,(ques_hn,ques_cn) = self.ques_rnn(input_q.transpose(0,1).float(),(ques_h0,ques_c0))

                video_fuse = self.video_linear(video_cn)
                ques_fuse = self.ques_linear(ques_cn)

                fuse = video_fuse*ques_fuse

                decoder_input = self.decoder_linear(fuse)

                answer_train = []
                y = y.float()
                y_mask = y_mask.float()
                loss = 0.0
                y_mask_sum = 0

                for i in range(self.max_n_a_words):
                        if i==0:
                                current_emb = decoder_input
                        else:
                                current_emb = self.word_to_lstm_linear(ans_vec[:,i-1,:].float())

                        #lstm
                        h,c = self.decoder_cell(current_emb,(decoder_h0,decoder_c0))

                        #ground truth
                        #onehot_labels = one_hot(y[:,i],t.zeros(batch_size,self.n_words))
                        logit_words = self.embed_word_linear(h)
                        _,max_prob_word = t.topk(logit_words,1)
                        answer_train.append(max_prob_word)
                        logit_words = F.log_softmax(logit_words.squeeze()).float()
                        #cross_entropy = t.nn.CrossEntropyLoss(reduce=False)(t.mul(logit_words.squeeze().float(),y_mask[:,i].unsqueeze(1)),t.mul(y[:,i].float(),y_mask[:,i].unsqueeze(1)))
                        cross_entropy = t.sum(-y[:,i].unsqueeze(1)*logit_words,1)*y_mask[:,i]
                        y_mask_sum += t.sum(y_mask[:,i])
                        current_loss = t.mean(cross_entropy)
                        loss += current_loss

                loss = loss/y_mask_sum
                return answer_train,loss
