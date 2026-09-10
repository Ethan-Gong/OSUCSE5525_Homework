# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        # The same Indexer is shared by every sentence. It gives each textual
        # feature (for example, "Unigram=excellent") one stable integer id.
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """Returns a sparse bag-of-words vector for one tokenized sentence."""
        features = Counter()
        for word in sentence:
            # Prefixing the word makes this feature distinct from a future
            # bigram feature that might contain the same text.
            feature_name = "Unigram=" + word
            feature_index = self.indexer.add_and_get_index(feature_name, add=add_to_indexer)

            # At dev/test time, unseen words return -1 and are intentionally
            # ignored: the model has no learned weight for them.
            if feature_index != -1:
                features[feature_index] += 1
        return features


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        # Never grow the vocabulary while predicting. In particular, dev and
        # blind-test words that were absent from training are ignored.
        features = self.feat_extractor.extract_features(ex_words, add_to_indexer=False)
        score = sum(self.weights[feature_index] * value
                    for feature_index, value in features.items())

        # sigmoid(score) is at least 0.5 exactly when score is nonnegative.
        return 1 if score >= 0 else 0


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor,
                              learning_rate: float = 0.1, num_epochs: int = 10) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # First build the training vocabulary and cache each sparse feature vector.
    # This gives the weight vector its final size before SGD begins.
    train_features = [feat_extractor.extract_features(ex.words, add_to_indexer=True)
                      for ex in train_exs]
    weights = np.zeros(len(feat_extractor.get_indexer()))

    for epoch in range(num_epochs):
        example_indices = list(range(len(train_exs)))
        random.shuffle(example_indices)

        for example_index in example_indices:
            features = train_features[example_index]
            label = train_exs[example_index].label
            score = sum(weights[feature_index] * value
                        for feature_index, value in features.items())

            # A numerically stable sigmoid implementation.
            if score >= 0:
                probability_positive = 1.0 / (1.0 + np.exp(-score))
            else:
                exp_score = np.exp(score)
                probability_positive = exp_score / (1.0 + exp_score)

            # Gradient ascent on the log likelihood of this example:
            # w <- w + learning_rate * (gold - predicted_probability) * x
            error = label - probability_positive
            for feature_index, value in features.items():
                weights[feature_index] += learning_rate * error * value

    return LogisticRegressionClassifier(weights, feat_extractor)


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor, args.lr, args.num_epochs)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        self.network = network
        self.word_embeddings = word_embeddings


    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        word_index = self.word_embeddings.word_indexer
        unk_id = word_index.index_of("UNK")
        word_ids = []
        for word in ex_words:
            word_id = word_index.index_of(word)
            if word_id == -1:
                word_id = unk_id
            word_ids.append(word_id)
        if len(word_ids) == 0:
            word_ids.append(unk_id)
        x = torch.tensor(word_ids, dtype=torch.long)
        self.network.eval()
        with torch.no_grad():
            output = self.network(x)
            predicted_label = torch.argmax(output).item()
        return predicted_label
    


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    num_classes = 2
    model = DAN(word_embeddings, args.hidden_size, num_classes)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    word_index= word_embeddings.word_indexer
    unk_id = word_index.index_of("UNK")
    for epoch in range(args.num_epochs):
        model.train()
        #['I', 'love', 'this', 'movie'] --> ['love', 'this', 'I', 'movie']
        random.shuffle(train_exs)
        #ex: ['I', 'love', 'this', 'movie']
        total_loss = 0.0
        for ex in train_exs:
            word_ids = []
            # ['I', 'love', 'this', 'movie'] --> ['I']
            for word in ex.words:
                #Assuming the word "I" in the glove.6B.50d-relativized.txt, then return word_id = 23
                word_id = word_index.index_of(word)
                #if not found, return the index of "UNK" in the glove.6B.50d-relativized.txt, which is 1
                if word_id == -1:
                    word_id = unk_id
                #['I', 'love', 'this', 'movie'] --> [23, 45, 67, 89]:word_ids is a list of word indices corresponding to the words in the example
                word_ids.append(word_id)
            x = torch.tensor(word_ids, dtype=torch.long)
            y = torch.tensor([ex.label], dtype=torch.long)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_function(output.unsqueeze(0), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_exs)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {average_loss:.10f}")
    return NeuralSentimentClassifier(model, word_embeddings)
    

class DAN(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings, hidden_size: int, num_classes: int):
        super(DAN, self).__init__()
        # Define the layer of the embedding layer
        # The embedding Layer is e(Wi),embedding_size is the size of the embedding vector, vocab_size is the size of the vocabulary,For example, embedding_size=50 in glove.6B.50d-relativized.txt, vocab_size is the number of words in the vocabulary, which can be obtained from the word_indexer in WordEmbeddings
        embedding_size = word_embeddings.get_embedding_length()
        self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=True)
        #g = f(W1 * e(Wi) + b1)
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        #v = ReLU(g)
        self.Relu = nn.ReLU()
        #y = (W2 * v + b2)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        #x is a [inp]-sized tensor of input data
        #embedding layer
        embedding = self.embedding(x)  
        #average the embeddings
        avg_words_emedding = torch.mean(embedding, dim=0) 
        #fully connected layer 1
        hidden_layer = self.fc1(avg_words_emedding)  
        #ReLU activation
        hidden_layer = self.Relu(hidden_layer)  
        hidden_layer = self.dropout(hidden_layer)
        #fully connected layer 2
        output = self.fc2(hidden_layer)
        # an number_classes sized tensor of data  
        return output
        
        