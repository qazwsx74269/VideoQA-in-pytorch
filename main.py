#coding:utf8
from config import opt
import os
import torch as t
import models
from visualizer import Visualizer
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import MyDataset
from torchnet import meter
from tqdm import tqdm
import time
import numpy as np
import pickle as pkl
import ipdb


def train(**kwargs):
        with open('/home/szh/mm_2018/data/tsn_score/caption/worddict.pkl','rb') as f1:
                word2index = pkl.load(f1)
        index2word = dict()
        for key in word2index.keys():
                index2word[word2index[key]] = key
        opt.parse(kwargs)
        vis = Visualizer(opt.env)
        model = getattr(models,opt.model)(opt.data_params,opt.model_params)
        if t.cuda.is_available:
                model.cuda()

        train_data = MyDataset(opt.data_params,opt.data_params['train_json'])
        train_dataloader = DataLoader(train_data,opt.data_params['batch_size'],shuffle=True,num_workers=1)
        #ipdb.set_trace()
        lr = opt.train_params['learning_rate']
        optimizer = t.optim.Adam(model.parameters(),lr=lr,weight_decay=opt.train_params['weight_decay'])
        loss_meter = meter.AverageValueMeter()

        print('Trainnning begins......')
        print ('****************************')
        print ('Trainning datetime:', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print ('Trainning params')
        print (opt.train_params)
        print ('Model params')
        print (opt.model_params)
        print ('Data params')
        print (opt.data_params)
        print ('****************************')

        for epoch in range(opt.train_params['max_epoches']):
                loss_meter.reset()

                # type_count = np.zeros(opt.data_params['n_types'], dtype=float)
                # wups_count = np.zeros(opt.data_params['n_types'], dtype=float)
                # wups_count2 = np.zeros(opt.data_params['n_types'], dtype=float)
                # bleu1_count = np.zeros(opt.data_params['n_types'], dtype=float)

                for ii,(img_frame_vecs, img_frame_n, ques_vecs, ques_n, ques_word, ans_vecs, ans_n, ans_word, type_vec,mask_matrix) in enumerate(tqdm(train_dataloader)):
                        if ans_vecs is None:
                                break
                        ques_vecs,ques_n,img_frame_vecs,img_frame_n,ans_word,ans_vecs,mask_matrix = Variable(ques_vecs.cuda()),Variable(ques_n.cuda()),\
                                Variable(img_frame_vecs.cuda()),Variable(img_frame_n.cuda()),\
                                Variable(ans_word.cuda()),Variable(ans_vecs.cuda()),Variable(mask_matrix.cuda())
                        #mask_matrix =np.zeros([np.shape(ans_n)[0], opt.data_params['max_n_a_words']], np.int32)
                        #nonzeros = map(lambda x: (x != 0).sum()+1, ans_word)
                        #nonzeros = np.zeros([np.shape(ans_n)[0]],np.int32)
                        #for i in range(np.shape(ans_n)[0]):
                        #     nonzero = list(filter(lambda x: x!=0,ans_word[i].cpu().data.numpy().tolist()))
                        #     print(nonzero)
                        #     nonzeros[i] = len(nonzero)
                        #print(nonzeros)
                        #for ind, row in enumerate(mask_matrix):
                        #        row[:nonzeros[ind]] = 1
                        #y_mask = Variable(t.from_numpy(mask_matrix).cuda().float(),requires_grad=False)
                        optimizer.zero_grad()
                        output,loss = model(ques_vecs,ques_n,img_frame_vecs,img_frame_n,ans_word,mask_matrix,ans_vecs,opt.data_params['batch_size'])
                        loss.backward()
                        optimizer.step()
                        loss_meter.add(loss.data[0])
                print(loss_meter.value()[0])
                print()


if __name__=='__main__':
        import fire
        fire.Fire()

