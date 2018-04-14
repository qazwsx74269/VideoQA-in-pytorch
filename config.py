#coding:utf8
import warnings

class DefaultConfig(object):
	env = 'default'
	model = 'EVQA'

	train_params = {
	'learning_rate': 1e-3,
	'lr_decay_n_iters': 2000,
	'lr_decay_rate': 0.8,
	'max_epoches': 1000,
	'early_stopping': 10,
	'cache_dir': './model/tsn/evqa/',
	'display_batch_interval': 20,
	'summary_interval': 10,
	'evaluate_interval': 5,
	'saving_interval': 1000,
	'epoch_reshuffle': True,
	'weight_decay':1e-4
	}
	model_params = {
		'use_q_len': True,
		'model_name': 'qvec_all_att_model',
		'lstm_dim': 384,
		'ref_dim':300,
		'ques_embed_dim': 100, # original: 300
		'attention_dim': 256,
		'regularization_beta': 1e-7,
		'use_notifier': True,
		'dropout_prob': 0.6,
		'boundary_dim': 128,
		'decode_dim': 256
	}
	data_params = {
		# general
		'dataset': 'tsn',
		'batch_size': 64,
		'n_types': 5,
		'n_words': 9611,                # change all time

		# feature
		'use_frame': True,
		'input_video_dim': 404,
		'max_n_frames': 240,

		# question answer
		'max_n_q_words': 20,
		'max_n_a_words': 10,
		'use_qvec': True,
		'input_ques_dim': 300,

		# path
		'wordvec_path': '/home/szh/mm_2018/data/tsn_score/caption/wordvec.npy',
		'word_dict_path': '/home/szh/mm_2018/data/tsn_score/caption/worddict.pkl',
		'train_json': '/home/szh/mm_2018/data/tsn_score/train_clean.json',
		'val_json': '/home/szh/mm_2018/data/tsn_score/val_clean.json',
		'test_json': '/home/szh/mm_2018/data/tsn_score/test_clean.json',
		'feat_path': '/home/szh/mm_2018/data/tsn_score/feat/',
		'flow_path': '/home/szh/mm_2018/data/tsn_score/flow/'
	}

	def parse(self,kwargs):
		'''
		update parameters according to kwargs
		'''
		for k,v in kwargs.items():
			if not hasattr(self,k):
				warnings.warn("Warning: opt has not attribute %s"%k)

		print("user config:")
		for k,v in self.__class__.__dict__.items():
			if not k.startswith('__'):
				print(k,getattr(self,k))

opt = DefaultConfig()
