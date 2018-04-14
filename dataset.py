#coding:utf8
import json
import pickle as pkl
import numpy as np
import random
import os
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
import ipdb

stopwords = ['.']
#ipdb.set_trace()
def load_file(filename):
        with open(filename,'rb') as f1:
                return pkl.load(f1)

def load_json(filename):
        with open(filename) as f1:
                return json.load(f1)

class MyDataset(Dataset):
        def __init__(self,params,data_file):
                # general
                self.dataset = params['dataset']
                self.feature_path = params['feat_path']
                self.flow_path = params['flow_path']
                self.params = params
                self.max_batch_size = params['batch_size']

                # dataset
                self.all_word_vec = np.load(params['wordvec_path'])
                self.word_dict = load_file(params['word_dict_path'])

                self.key_file = load_json(data_file)
                #print("hello")
                #print(type(self.key_file))
                # frame / motion / question
                self.use_frame = params['use_frame']
                if self.use_frame:
                        self.max_n_frames = params['max_n_frames']
                        self.v_dim = params['input_video_dim']
                self.max_q_words = params['max_n_q_words']
                self.max_a_words = params['max_n_a_words']
                self.use_qvec = params['use_qvec']
                if self.use_qvec:
                        self.q_dim = params['input_ques_dim']

        def __getitem__(self,index):
                #ipdb.set_trace()
                img_frame_vecs = np.zeros((self.max_n_frames, self.v_dim), dtype=float)
                img_frame_n = np.zeros((1), dtype=int)

                ques_vecs = np.zeros((self.max_q_words, self.q_dim), dtype=float)
                ques_word = np.zeros((self.max_q_words), dtype=int)
                ques_n = np.zeros((1), dtype=int)

                ans_vecs = np.zeros((self.max_a_words, self.q_dim), dtype=float)
                ans_word = np.zeros((self.max_a_words), dtype=int)
                ans_n = np.zeros((1), dtype=int)

                type_vec = np.zeros((1), dtype=int)
                item = self.key_file[index]
                vid = item[0][2:]
                mask_matrix = np.zeros([self.max_a_words],np.int32)

                #print(vid)
                if self.use_frame:
                        #if not os.path.exists(self.feature_path + '%s.h5' % vid):
                        #        continue
                        with h5py.File(self.feature_path + '%s.h5' % vid, 'r') as hf:
                                fg = np.asarray(hf['fg'])
                                bg = np.asarray(hf['bg'])
                                feat = np.hstack([fg, bg])
                        with h5py.File(self.flow_path + '%s.h5' % vid, 'r') as hf:
                                fg2 = np.asarray(hf['fg'])
                                bg2 = np.asarray(hf['bg'])
                                feat2 = np.hstack([fg2, bg2])
                        feat = np.hstack([feat, feat2]) # [frame, 404]
                        if len(feat) > self.max_n_frames:
                                index = np.linspace(0, len(feat)-1, self.max_n_frames).astype(np.int32)
                                feat = feat[index, :]
                        n_frames = len(feat)
                        img_frame_vecs[:n_frames, :] = feat
                        img_frame_n = n_frames
                #print(index)
                ques = item[2]
                #print(type(ques))
                ques = ques.split()
                ques = [word.lower() for word in ques if word not in stopwords and word != '']
                ques = [self.word_dict[word] if word in self.word_dict else 0 for word in ques]
                vector = self.all_word_vec[ques]
                ques_n = min(len(ques), self.max_q_words)
                ques_word[:ques_n] = ques[:ques_n]
                ques_vecs[:ques_n, :] = vector[:ques_n, :]

                # answer
                #print(self.key_file[index][3])
                ans = item[3]
                ans = ans.split()
                #print(1)
                #print(ans)
                ans = [word.lower() for word in ans if word not in stopwords and word != '' and word != 'EOS']
                #print(2)
                ans += ['EOS']
                #print(ans)
                ans = [self.word_dict[word] if word in self.word_dict else 0 for word in ans]
                #print(3)
                #print(ans)
                #print(type(ans))
                if len(ans) > self.max_a_words:
                        #print("excess!!!")
                        #print(ans[:self.max_a_words-1])
                        ans = ans[:self.max_a_words-1]+ans[-1:]
                        #print('asd')
                        #print(ans)
                mask_matrix[[mi for mi,v in enumerate(ans) if v!=0]] = 1
                #print("ms")
                #print(mask_matrix)
                vector = self.all_word_vec[ans]
                #print(4)
                #print(ans)
                ans_n = min(len(ans), self.max_a_words)
                ans_word[:ans_n] = ans[:ans_n]
                ans_vecs[:ans_n, :] = vector[:ans_n, :]

                type_vec = item[4]

                return img_frame_vecs, img_frame_n, ques_vecs, ques_n, ques_word, ans_vecs, ans_n, ans_word, type_vec,mask_matrix

        def __len__(self):
            return len(os.listdir(self.feature_path))

