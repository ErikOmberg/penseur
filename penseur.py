import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import os, skipthoughts, penseur_utils

class Penseur:

	def __init__(self, model_name=''):
		self.loaded_custom_model = False
		if model_name == '':
			print ('Loading BookCorpus encoding model...')
			self.model = skipthoughts.load_model()
			self.sentences = None
			self.vectors = None
		else:
			print ('Loading custom encoding model: ' + model_name)
			self.loaded_custom_model = True
			self.model = penseur_utils.load_encoder(model_name)
			self.sentences = pickle.load(open('data/' + model_name + '_sen.p', 'r'))
			self.encode(self.sentences, verbose=True)
		self.analogy_vector = None
		self.word_table = None

	# Loads both an encoding file and its sentences from disc
	def load(self, filename):
		self.vectors = np.load('data/' + filename + '_encoder.np', 'r')
		self.sentences = pickle.load(open('data/' + filename + '_sen.p', 'rb'))

	# Encodes a list of sentences
	def encode(self, sentences):
		self.sentences = sentences
		if self.loaded_custom_model:
			self.vectors = penseur_utils.encode(self.model, sentences)
		else:
			self.vectors = skipthoughts.encode(self.model, sentences)

	# Saves a set of encodings and the corresponding sentences to disc
	def save(self, filename):
		if not os.path.exists('data/'):
			os.makedirs('data')
		np.save(open('data/' + filename + '_encoder.np', 'wb'), self.vectors)
		pickle.dump(self.sentences, open('data/' + filename + '_sen.p', 'wb'))

	# Returns a list of the sentences closest to the input sentence
	def get_closest_sentences(self, query_sentence, num_results=5):
		return skipthoughts.nn(self.model, self.sentences, self.vectors, query_sentence, self.loaded_custom_model, num_results)

	# Returns a list of the words closest to the input word
	def get_closest_words(self, query_word, num_results=5):
		if self.loaded_custom_model:
			if self.word_table is None:
				self.word_table = skipthoughts.word_features(self.model['table'])
			return skipthoughts.nn_words(self.model['table'], self.word_table, query_word, num_results)
		else:
			if self.word_table is None:
				self.word_table = skipthoughts.word_features(self.model['btable'])
			return skipthoughts.nn_words(self.model['btable'], self.word_table, query_word, num_results)

	# Returns the vector of a query sentence within the current embedding space
	def get_vector(self, query_sentence):
		return skipthoughts.vector(self.model, self.sentences, self.vectors, query_sentence, self.loaded_custom_model)

	# Returns a simple distance between sentences
	def get_distance(self, query_sentence1, query_sentence2):
		v1 = self.get_vector(query_sentence1)
		v2 = self.get_vector(query_sentence2)
		return (abs(v1) - abs(v2)).sum()

	# Returns the sentence of a query vector
	def get_sentence(self, query_vector):
		return skipthoughts.sentence(self.model, self.sentences, self.vectors, query_vector)

	# Loads pairs of sentences (ie questions and answers) from disc
	def load_pairs(self, filename):
		with open(filename + '.txt', 'r') as f:
			s = f.readlines()
		av = []
		for i in xrange(0, len(s), 3):
			cv = self.get_vector(s[i+1].replace('\n', '')) - self.get_vector(s[i].replace('\n', ''))
			av.append(cv)
		return np.average(np.array(av), axis=0)

	# Returns the response using the average vector from load_pairs input file
	def analogy(self, query_sentence, filename='q&a_pairs'):
		if self.analogy_vector is None:
			if os.path.isfile(filename + '.np'):
				self.analogy_vector = np.load(filename + '.np', 'r')
			else:
				self.load_and_save_analogy_file(filename)
		try:
			return self.get_sentence(self.get_vector(query_sentence) + self.analogy_vector)
		except:
			self.load_and_save_analogy_file(filename)
			return self.get_sentence(self.get_vector(query_sentence) + self.analogy_vector)

	def load_and_save_analogy_file(self, filename='q&a_pairs'):
		self.analogy_vector = self.load_pairs(filename)
		np.save(open(filename + '.np', 'w'), self.analogy_vector)

	# Displays the plot of the sentence encodings after PCA (to 2D)
	def display_PCA_plot(self):
		try:
			plot_data = self.PCA(np.squeeze(np.array(self.vectors)))
			for i, v in enumerate(plot_data):
				plt.scatter(v[0], v[1])
				plt.annotate(self.sentences[i], (v[0], v[1]))
			plt.title("PCA plot")
			plt.show()
		except:
			print("Not enough memory; corpus too large for this function")

	# Performs PCA on the sentence encodings
	def PCA(self, data, rescaled_dims=2):
		m, n = data.shape

		# Center around the mean
		plot_data = data - data.mean(axis=0)

		# Covariance matrix
		r = np.cov(plot_data, rowvar=False)

		# Get eigenvals, eigenvectors
		evals, evecs = np.linalg.eigh(r)

		# Sort eigevalue decreasing order
		idx = np.argsort(evals)[::-1]
		evecs = evecs[:,idx]

		# Sort eigenvects by same index
		evals = evals[idx]

		# Select first n eigenvectors
		evecs = evecs[:, :rescaled_dims]

		return np.dot(evecs.T, plot_data.T).T

	# Flattens vectors for PCA
	def flatten(self, data, x_vector, y_vector):
		vectors = np.array([x_vector, y_vector])
		return np.dot(vectors, data.T).T

	# Displays the sentence encodings after PCA with axis constraints
	def display_constrained_plot(self, x_axis_sentences, y_axis_sentences):
		if len(x_axis_sentences) != 2 or len(y_axis_sentences) != 2:
			sys.exit("Displaying PCA plot with constraints: expected 4 sentences. Got " + \
			str(len(x_axis_sentences)) + ' and ' + str(len(y_axis_sentences)))

		x_axis = self.get_vector(x_axis_sentences[0]) - self.get_vector(x_axis_sentences[1])
		y_axis = self.get_vector(y_axis_sentences[0]) - self.get_vector(y_axis_sentences[1])

		data = []
		for s in self.sentences:
			data.append(self.get_vector(s))

		flattened_data = self.flatten(np.squeeze(np.array(data)), x_axis, y_axis)
		plt.xlabel = ('[' + x_axis_sentences[0][:20] + '...] - [' + x_axis_sentences[1][:20] + '...]')
		plt.ylabel = ('[' + y_axis_sentences[0][:20] + '...] - [' + y_axis_sentences[1][:20] + '...]')

		for i, v in enumerate(np.squeeze(flattened_data)):
			plt.scatter(v[0], v[1])
			plt.annotate(self.sentences[i], (v[0], v[1]))

		plt.title("Flattened data")
		plt.show()



