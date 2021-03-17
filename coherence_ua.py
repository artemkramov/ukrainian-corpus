import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Input, Embedding, LSTM
import ufal.udpipe
from models.transformer.transformer_model import TransformerCoherence
from models.transformer.utils import *
import numpy as np


class CoherenceModel:

  nn = None

  document = None
  
  embedder = None

  ud_model = None
  
  def set_embedder(self, _embedder):
    self.embedder = _embedder

  def __init__(self):
    self.ud_model = UniversalDependencyModel("bin/ukrainian-iu-ud-2.3-181115.udpipe")
    num_layers = 1
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    embedding_size = 1024

    self.nn = TransformerCoherence(num_layers, embedding_size, num_heads, dff, dropout_rate)
    self.nn.load_weights("bin/transformer/chk-2-1")

    pass

  def set_nn(self, _nn):
    self.nn = _nn

  def set_document(self, _document):
    self.document = _document

  def mask_sample(self, sample):
    return tf.keras.preprocessing.sequence.pad_sequences(sample, MAX_WORDS, dtype='float32')

  def prepare_input(self, sample):
    return {'input1': sample[0], 'input2': sample[1], 'input3': sample[2]}

  def get_prediction_series(self, text):
    sentences = self.ud_model.get_tokens(text)
    document = self.embedder.sents2elmo(sentences)

    clique_length = 3

    counter = clique_length - 1

    samples = {'input1': [], 'input2': [], 'input3': []}

    if len(document) < 3:
      return tf.constant([0])

    while counter < len(document):
      sentences = []
      for i in range(counter - clique_length, counter):
        sentences.append(document[i])
      counter += 1

      item = self.prepare_input(self.mask_sample(sentences))
      samples['input1'].append(item['input1'])
      samples['input2'].append(item['input2'])
      samples['input3'].append(item['input3'])

    samples['input1'] = np.array(samples['input1'])
    samples['input2'] = np.array(samples['input2'])
    samples['input3'] = np.array(samples['input3'])

    predictions = self.nn(samples, False).numpy()

    return predictions

  def evaluate_coherence_as_product(self, text):
    
    predictions = self.get_prediction_series(text)
    return np.multiply.reduce(predictions)
    #return np.mean(predictions)

  def evaluate_coherence_using_threshold(self, text, threshold=0.5):
    predictions = self.get_prediction_series(text)

    predictions = [1 if output > threshold else 0 for output in predictions]
    return np.sum(predictions) / len(predictions)


class UniversalDependencyModel:

  # udpipe compiled model
  model = None

  def __init__(self, path):
    # Load model by the given path
    self.model = ufal.udpipe.Model.load(path)
    if not self.model:
      raise Exception("Cannot load model by the given path: %s" % path)

  def parse(self, sentence):
    self.model.parse(sentence, self.model.DEFAULT)

  def tokenize(self, text):
    """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
    tokenizer = self.model.newTokenizer(self.model.DEFAULT)
    if not tokenizer:
      raise Exception("The model does not have a tokenizer")
    return self._read(text, tokenizer)

  def read(self, text, in_format):
    """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
    input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
    if not input_format:
      raise Exception("Cannot create input format '%s'" % in_format)
    return self._read(text, input_format)

  def _read(self, text, input_format):
    input_format.setText(text)
    error = ufal.udpipe.ProcessingError()
    sentences = []

    sentence = ufal.udpipe.Sentence()
    while input_format.nextSentence(sentence, error):
      sentences.append(sentence)
      sentence = ufal.udpipe.Sentence()
    if error.occurred():
      raise Exception(error.message)

    return sentences

  def tag(self, sentence):
    """Tag the given ufal.udpipe.Sentence (inplace)."""
    self.model.tag(sentence, self.model.DEFAULT)

  def get_tokens(self, input_text):

    items = []

    sentences = self.tokenize(input_text)
    # Loop through each sentence, split them into word, perform lemmatization
    for s in sentences:

      # Parse each sentence to retrieve features
      self.tag(s)
      self.parse(s)

      # Collect all lemmas into one list
      i = 0
      words = []
      while i < len(s.words):
        if s.words[i].id != 0:
          words.append(s.words[i].form)
        i += 1
      items.append(words)
    return items

  def write(self, sentences, out_format):
    """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

    output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
    output = ''
    for sentence in sentences:
      output += output_format.writeSentence(sentence)
    output += output_format.finishDocument()

    return output

