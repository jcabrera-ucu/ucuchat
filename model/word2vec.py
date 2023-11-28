import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import sys

from sklearn.manifold import TSNE

EMBED_DIMENSION = 300
EMBED_MAX_NORM = 1


def load_model_and_vocab(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f'{model_dir}/model.pt', map_location=device)
    vocab = torch.load(f'{model_dir}/vocab.pt')

    return model, vocab


class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size
        )

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_DIMENSION
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size
        )

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = self.linear(x)
        return x


class Word2VecModel:
    """
    Class to storage a skip-gram or cbow word2vec model.
    """

    def __init__(self, model_dir):
        model, vocab = load_model_and_vocab(model_dir)
        self.model = model
        self.vocab = vocab

    def get_embeddings(self):
        # embedding from first model layer
        embeddings = list(self.model.parameters())[0]
        embeddings = embeddings.cpu().detach().numpy()

        # normalization
        norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
        norms = np.reshape(norms, (len(norms), 1))
        embeddings_norm = embeddings / norms

        return embeddings, embeddings_norm

    def get_similar_words(self, word, number_of_words=10):
        word_id = self.vocab[word]
        if word_id == 0:
            return

        embeddings_norm = self.get_embeddings()[1]
        word_vec = embeddings_norm[word_id]
        word_vec = np.reshape(word_vec, (len(word_vec), 1))
        dists = np.matmul(embeddings_norm, word_vec).flatten()
        number_of_words_ids = np.argsort(-dists)[1: number_of_words + 1]

        number_of_words_dict = {}
        for sim_word_id in number_of_words_ids:
            sim_word = self.vocab.lookup_token(sim_word_id)
            number_of_words_dict[sim_word] = dists[sim_word_id]
        return number_of_words_dict

    def get_similar_words_sentence(self, sentence, number_of_words=10):
        words = sentence.lower().split()
        embeddings = self.get_embeddings()[0]
        sentence_vec = None

        for word in words:
            word_id = self.vocab[word]
            if word_id == 0:
                continue
            if sentence_vec is None:
                sentence_vec = embeddings[word_id]
            else:
                sentence_vec = sentence_vec + embeddings[word_id]

        embeddings_norm = self.get_embeddings()[1]
        # word_vec = embeddings_norm[word_id]
        word_vec = np.reshape(sentence_vec, (len(sentence_vec), 1))
        dists = np.matmul(embeddings_norm, word_vec).flatten()
        number_of_words_ids = np.argsort(-dists)[1: number_of_words + 1]

        number_of_words_dict = {}
        for sim_word_id in number_of_words_ids:
            sim_word = self.vocab.lookup_token(sim_word_id)
            number_of_words_dict[sim_word] = dists[sim_word_id]
        return number_of_words_dict

    def get_text_vector(self, text):
        embeddings = self.get_embeddings()[0]
        words = text.lower().split()
        vec = None

        for word in words:
            word_id = self.vocab[word]
            if word_id == 0:
                continue
            if vec is None:
                vec = embeddings[word_id]
            else:
                vec = vec + embeddings[word_id]
        return vec

    def get_distance_sentences(self, sentence, sentence2):
        # embeddings = self.get_embeddings()[0]

        vec1 = self.get_text_vector(sentence)
        vec2 = self.get_text_vector(sentence2)
                
        if vec1 is not None and vec2 is not None:
            cosine_similarity = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
        else:
            cosine_similarity = 0.0

        return cosine_similarity

    def vector_equations(self):
        embeddings = self.get_embeddings()[0]
        emb1 = embeddings[self.vocab["king"]]
        emb2 = embeddings[self.vocab["man"]]
        emb3 = embeddings[self.vocab["woman"]]

        emb4 = emb1 - emb2 + emb3
        emb4_norm = (emb4 ** 2).sum() ** (1 / 2)
        emb4 = emb4 / emb4_norm

        emb4 = np.reshape(emb4, (len(emb4), 1))
        dists = np.matmul(embeddings[1], emb4).flatten()

        top5 = np.argsort(-dists)[:5]

        for word_id in top5:
            print("{}: {:.3f}".format(self.vocab.lookup_token(word_id), dists[word_id]))
