# when you run this script, add a THEANO-FLAG command to the front:
# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python

import sys, os
import pickle as pickle

def train_encoder(name_of_data, sentences, max_epochs=5, save_frequency=1000):
	if not os.path.exists('data/'):
		os.makedirs('data')
	sys.path.insert(0, 'training/')
	import vocab
	worddict, wordcount = vocab.build_dictionary(sentences)
	vocab.save_dictionary(worddict, wordcount, 'data/' + name_of_data + '_dictionary.pkl')
	pickle.dump(sentences, open('data/' + name_of_data + '_sen.p', 'wb'))
	with open('training/train.py', 'r') as f:
		text = f.read()
		text = text.replace('max_epochs=5', 'max_epochs=' + str(max_epochs))
		text = text.replace('saveto=\'/u/rkiros/research/semhash/models/toy.npz\'',\
			'saveto=\'data/' + name_of_data + '_encoder.npz\'')
		text = text.replace('dictionary=\'/ais/gobi3/u/rkiros/bookgen/book_dictionary_large.pkl\'',\
			'dictionary=\'data/' + name_of_data + '_dictionary.pkl\'')
		text = text.replace('n_words=20000', 'n_words=' + str(len(wordcount.keys())))
		text = text.replace('saveFreq=1000', 'saveFreq=' + str(save_frequency))
		g = open('training/train_temp.py', 'w')
		g.write(text)
		g.close()

	import train_temp
	train_temp.trainer(sentences)

def load_encoder(model_name):
	sys.path.insert(0, 'training/')
	import tools
	return tools.load_model('data/' + model_name + '_encoder.npz', 'data/' + model_name + '_dictionary.pkl',\
		'data/GoogleNews-vectors-negative300.bin')

def encode(encoder, sentences, verbose=False):
	sys.path.insert(0, 'training/')
	import tools
	return tools.encode(encoder, sentences)

def train_decoder(name_of_data, sentences, model, max_epochs=5, save_frequency=1000):
	if not os.path.exists('data/'):
		os.makedirs('data')
	sys.path.insert(0, 'decoding/')
	import vocab
	worddict, wordcount = vocab.build_dictionary(sentences)
	vocab.save_dictionary(worddict, wordcount, 'data/' + name_of_data + '_dictionary.pkl')
	with open('decoding/train.py', 'r') as f:
		text = f.read()
		text = text.replace('max_epochs=5', 'max_epochs=' + str(max_epochs))
		text = text.replace('saveto=\'/u/rkiros/research/semhash/models/toy.npz\'',\
			'saveto=\'data/' + name_of_data + '_decoder.npz\'')
		text = text.replace('dictionary=\'/ais/gobi3/u/rkiros/bookgen/book_dictionary_large.pkl\'',\
			'dictionary=\'data/' + name_of_data + '_dictionary.pkl\'')
		text = text.replace('n_words=40000', 'n_words=' + str(len(wordcount.keys())))
		text = text.replace('saveFreq=1000', 'saveFreq=' + str(save_frequency))
		g = open('decoding/train_temp.py', 'w')
		g.write(text)
		g.close()

	import train_temp
	return train_temp.trainer(sentences, sentences, model)

def load_decoder(decoder_name):
	sys.path.insert(0, 'decoding/')
	import tools
	return tools.load_model('data/' + decoder_name + '_decoder.npz', 'data/' + decoder_name + '_dictionary.pkl')

def decode(decoder, vector, num_results=1):
	sys.path.insert(0, 'decoding/')
	import tools
	sentences = tools.run_sampler(decoder, vector, beam_width=num_results)
	if num_results == 1:
		return sentences[0]
	return sentences


