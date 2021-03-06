import numpy as np
from ..embedding.semantic_embedding import SemanticEmbedding
from ..embedding.scalar_embedding import ScalarEmbedding
from .action import PassAction, MergeAction


# Class which defines network policy
class Policy:
    # Neural network model (binary classifier)
    model = None

    model_semantic = None

    # Embedding to extract semantic data
    semantic_embedding = None

    # Embedding to extract scalar data
    scalar_embedding = None

    # Sieve to filter obvious pairs of mentions
    sieve = None

    # Threshold to decide if clusters are compatible to be merged
    PROB_THRESHOLD = 0.5

    transformers = None

    def __init__(self, model, transformers):
        self.model = model
        self.transformers = transformers
        self.load_embeddings()

    # Load embedding models
    def load_embeddings(self):
        self.semantic_embedding = SemanticEmbedding()
        self.scalar_embedding = ScalarEmbedding()

        # Load sieve
        # self.sieve = Sieve()

    # Evaluate ability for cluster merging
    def apply_train(self, cluster_mention, cluster_antecedent):

        cluster1, cluster2 = self.clusters_to_matrices(cluster_mention, cluster_antecedent)

        # Run neural network to predict
        prediction = self.model(cluster1, cluster2)

        return prediction

    # Evaluate ability for cluster merging
    def apply_test(self, cluster_mention, cluster_antecedent):
        vectors = []
        for mention1 in cluster_mention:
            for mention2 in cluster_antecedent:
                pair = [mention2, mention1]
                vector = self.scalar_embedding.pair2vec(pair)
                vector.append(self.semantic_embedding.get_pair_similarity(pair))
                vectors.append(vector)
        vectors = np.array(vectors)
        vectors = self.transformers.transform(vectors)
        # df = pd.DataFrame(vectors)
        # df.columns = ['is_pronoun1', 'tfidf1', 'is_pronoun2', 'tfidf2', 'is_dem2', 'count_of_words',
        #               'count_of_entities', 'is_lemma_equal', 'is_number_entities_same', 'is_gender_same',
        #               'is_proper_name', 'is_additional']
        # df[['count_of_words', 'count_of_entities']] = self.transformers.fit_transform(df)
        prediction_scalar = np.mean(self.model.predict_proba(vectors)[:, 1])
        cluster1, cluster2 = self.clusters_to_matrices(cluster_mention, cluster_antecedent)

        # Run neural network to predict
        prediction_network = self.model_semantic([cluster1, cluster2])[0][0]

        return prediction_network.numpy(), prediction_scalar

    # Evaluate ability for cluster merging
    def apply(self, cluster_mention, cluster_antecedent):

        cluster1, cluster2 = self.clusters_to_matrices(cluster_mention, cluster_antecedent)
        # Run neural network to predict
        prediction_network = self.model_semantic([cluster1, cluster2])[0][0]

        if prediction_network >= self.PROB_THRESHOLD:

            vectors = []
            for mention1 in cluster_mention:
                for mention2 in cluster_antecedent:
                    pair = [mention2, mention1]
                    vector = self.scalar_embedding.pair2vec(pair)
                    vector.append(self.semantic_embedding.get_pair_similarity(pair))
                    vectors.append(vector)
            vectors = np.array(vectors)
            vectors = self.transformers.transform(vectors)
            prediction_scalar = np.mean(self.model.predict_proba(vectors)[:, 1])

            if prediction_scalar > 0.1:
                return True

        return False

    # Transform pair of clusters to common matrix
    def clusters_to_matrices(self, cluster_mention, cluster_antecedent):
        # Form pair "mention"-"antecedent"

        cluster1 = np.expand_dims(self.semantic_embedding.cluster_to_matrix(cluster_mention), axis=0)
        cluster2 = np.expand_dims(self.semantic_embedding.cluster_to_matrix(cluster_antecedent), axis=0)

        return cluster1, cluster2

    # Preprocess document to fit features of embeddings
    def preprocess_document(self, mentions):
        tokens = []
        for mention in mentions:
            tokens.extend(mention.tokens)
        self.scalar_embedding.evaluate_tfidf(tokens)
        self.semantic_embedding.tokens = tokens
        self.semantic_embedding.clear_phrase_cache()

    # Form matrix of pairs of entities
    # from the entity links
    @staticmethod
    def get_matrix_from_pair(pair):

        # Init matrix
        pair_matrix = []

        # Loop through each pair and create all possible pair variants
        # Fetch each entity data by the link
        for first_mention in pair[0]:
            for second_mention in pair[1]:
                pair_matrix.append((first_mention, second_mention))
        return pair_matrix

    def get_scalar_matrix_from_pair(self, pair_matrix):
        return self.scalar_embedding.matrix2vec(pair_matrix)

    # Get semantic matrix from the given pair of entities
    def get_semantic_matrix_from_pair(self, pair_matrix):
        matrix = []

        # From the matrix of tokens
        # form the corresponding matrix of words
        for entity_pair in pair_matrix:

            # Words of each pair
            words = []
            for entity_tokens in entity_pair:

                # Words of each entity
                entity_words = []
                for token in entity_tokens.tokens:
                    entity_words.append(token)
                words.append(entity_words)
            matrix.append(words)

        return self.semantic_embedding.matrix2vec(matrix)


class ReferencePolicy:

    @staticmethod
    def apply(state: 'State', clusters_gold):
        if (state.current_mention_idx < len(
                clusters_gold)) and state.current_mention_idx > state.current_antecedent_idx:
            is_merge = True
            if clusters_gold[state.current_mention_idx] != '' and clusters_gold[state.current_antecedent_idx] != '':
                cluster_ant = state.get_siblings_of_mention(state.current_antecedent_idx)
                for cluster_ant_id in cluster_ant:
                    if clusters_gold[cluster_ant_id] != clusters_gold[state.current_mention_idx]:
                        is_merge = False
                        break
                if is_merge:
                    return MergeAction()
        return PassAction()