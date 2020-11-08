import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
import joblib
import configparser
from os import listdir
from os.path import isfile, join
import os
import dill
from typing import List
from models.searn.agent import Agent
from models.searn.policy import Policy, ReferencePolicy
from models.searn.mention import Mention, Word
import noun_phrase_ua
import logging


class CoreferenceUA:

    # Folder to save models during training
    folder_models = "bin"

    # Template to save models
    filename_template = "model_{0}"

    folder_conll = "test/conll"

    folder_texts = "test/texts"

    model = None

    model_semantic = None

    epoch = '-'

    transformers = None

    document = None

    current_folder = ""

    policy = None

    def __init__(self):

        logging.getLogger('elmoformanylangs').setLevel(logging.WARNING)

        # Load configuration data
        config = configparser.ConfigParser()
        config.read('config/train.ini')
        self.config = config

        self.current_folder = os.path.dirname(os.path.abspath(__file__))

        self.build_model()
        self.build_model_semantic(20)

        # Policy to learn
        self.policy = Policy(self.model, self.transformers)
        self.policy.model_semantic = self.model_semantic

    # Build model due to the configuration parameters
    def build_model_semantic(self, epoch_number=None):
        embedding_size = 1024 * 4
        lstm_units = 128
        dense_units = 128

        model = CoreferentClusterModel(embedding_size, lstm_units, dense_units)
        if not (epoch_number is None):
            subfolder = join(self.current_folder, self.folder_models, str(epoch_number))
            # Prepare filenames of the model
            filename_weights = self.filename_template.format(epoch_number)
            model.load_weights(join(subfolder, filename_weights))
            self.epoch = epoch_number
        self.model_semantic = model

    # Build model due to the configuration parameters
    def build_model(self):
        self.model = joblib.load(join(self.current_folder, self.folder_models, "extra_trees_clf_1.pkl"))
        self.transformers = dill.load(open(join(self.current_folder, self.folder_models, "transformers_1.pickle"), mode='rb'))

    def transform_token(self, token, counter):
        word = Word()
        word.RawText = token['word']
        word.WordOrder = counter
        word.PartOfSpeech = token['pos']
        word.Lemmatized = token['lemma']
        is_plural, gender = self.parse_tag(token['tag'])
        word.IsPlural = is_plural
        word.IsProperName = token['isProperName']
        word.IsHeadWord = token['isHeadWord']
        word.Gender = gender
        word.EntityID = token['groupID']
        word.RawTagString = token['tag']
        return word

    # Parse tag string (like Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing
    @staticmethod
    def parse_morphological_tag(tag_string):
        # Split by delimiter to separate each string
        morphology_strings = tag_string.split('|')
        morphology_attributes = []
        for morphology_string in morphology_strings:
            # Split each string to fetch attribute and its value
            morphology_attribute = morphology_string.split('=')
            morphology_attributes.append(morphology_attribute)
        return morphology_attributes

    # Fetch morphological feature by the given name
    def fetch_morphological_feature(self, tag_string, feature_name):
        morphology_attributes = self.parse_morphological_tag(tag_string)
        return [attribute_data[1] for attribute_data in morphology_attributes if attribute_data[0] == feature_name]

    # Parse tag string (like Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing
    # https://universaldependencies.org/u/feat/index.html
    def parse_tag(self, tag_string):
        morphology_attributes = self.parse_morphological_tag(tag_string)

        # Set initial data
        is_plural = False
        gender = None

        for morphology in morphology_attributes:

            # Extract gender
            if morphology[0] == 'Gender':
                gender = morphology[1]

            # Extract plurality
            if morphology[0] == 'Number' and morphology[1] == 'Plur':
                is_plural = True

        return is_plural, gender

    def set_text(self, text):
        summary = noun_phrase_ua.NLP().extract_entities(text)
        tokens = []
        counter = 0

        while counter < len(summary['tokens']):

            is_entity_group = False
            for entity_group in summary['entities']:
                if counter in entity_group:
                    is_entity_group = True
                    items = []
                    while counter <= entity_group[-1]:
                        items.append(self.transform_token(summary['tokens'][counter], counter))
                        counter += 1

                    mention = Mention(items)
                    mention.is_entity = True
                    tokens.append(mention)
                    break
            if not is_entity_group:

                mention = Mention([self.transform_token(summary['tokens'][counter], counter)])
                if summary['tokens'][counter]['isEntity']:
                    mention.is_entity = True
                tokens.append(mention)
                counter += 1

        self.document = tokens
        return self

    def extract_phrases(self):
        agent = Agent(self.document)
        agent.set_sieve()
        self.policy.preprocess_document(self.document)
        agent.move_to_end_state(self.policy)
        words_predicted, groups_predicted = agent.state_to_list(agent.states[-1])
        print("Predicted: " + " ".join(words_predicted))
        print("Coreferent pairs: " + str(groups_predicted))
        return agent.state_to_json(agent.states[-1])

    def run(self):

        # Load mentions from DB
        folder = 'dataset_2'
        config_training = self.config['TRAINING']
        files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

        # Policy to learn
        policy = Policy(self.model, self.transformers)
        reference_policy = ReferencePolicy()

        policy.model_semantic = self.model_semantic

        # Percentage of documents for training purpose
        training_split = float(config_training['training_split'])

        for filename in files:
            # Read documents
            handle = open(filename, 'rb')
            documents: List[List[Mention]] = pickle.load(handle)
            # documents = documents[:1]
            handle.close()

            # Calculate separator index to divide documents into 2 parts
            separator_index = int(training_split * len(documents)) + 1

            predict = []
            actual = []

            # separator_index = 2300

            print(len(documents))

            lines = []

            predictions = []

            for document_id, document in enumerate(documents[separator_index:]):
                print(document_id)
                agent = Agent(document)
                agent.set_gold_state(document)
                # agent.set_sieve()
                policy.preprocess_document(document)
                agent.move_to_end_state(policy)
                # print(agent.predictions)
                # predictions = agent.predictions

                conll_predict = agent.state_to_conll(agent.states[-1], document_id)
                conll_actual = agent.state_to_conll(agent.state_gold, document_id)
                predict.append(conll_predict)
                actual.append(conll_actual)

                words_predicted, groups_predicted = agent.state_to_list(agent.states[-1], document_id)
                words_actual, groups_actual = agent.state_to_list(agent.state_gold, document_id)

                lines.append("Actual: " + " ".join(words_actual))
                lines.append("Coreferent pairs: " + str(groups_actual))
                lines.append("Predicted: " + " ".join(words_predicted))
                lines.append("Coreferent pairs: " + str(groups_predicted))
                lines.append("\r\n")
                # self.save_file(conll_predict, document_id, False)
                # self.save_file(conll_actual, document_id, True)
                # print(agent.actions)'''

            file = self.epoch
            self.save_file(os.linesep.join(predict), file, False)
            self.save_file(os.linesep.join(actual), file, True)
            self.save_file_texts(lines, self.epoch)
            # with open('test/predictions_1.pkl', 'wb') as file:
            #     dill.dump(predictions, file)


class CoreferentClusterModel(Model):

    def __init__(self, embedding_size, lstm_units, dense_units, **kwargs):
        super(CoreferentClusterModel, self).__init__(**kwargs)

        self.bilstm = Bidirectional(
            LSTM(lstm_units, activation='tanh', recurrent_activation="sigmoid", input_shape=(None, embedding_size),
                 dtype=tf.float32))
        self.dense1 = Dense(dense_units)
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        cluster1 = self.bilstm(inputs[0])
        cluster2 = self.bilstm(inputs[1])

        cluster = tf.concat([cluster1, cluster2], axis=-1)
        x = self.dense1(cluster)

        return self.dense2(x)
