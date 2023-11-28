import sys
import time
import random
import json
import numpy as np


class Graph:
    def __init__(self, nlp, trained_model):
        self.nlp = nlp
        self.trained_model = trained_model
        self.graph = {}
        self.max_context = 1

    def load(self, graph):
        self.graph = graph['graph']
        self.max_context = graph['max_context']

    def as_dict(self):
        return {
            "graph": self.graph, 
            "max_context": self.max_context,
        }

    def sentences(self, initial_text, context):
        """
        Generator of words starting from `initial_text` with a
        sliding window of `context` words.
        """

        for w in self.posified_sentences(initial_text, context):
            yield w.split("::")[0]

    def sentences_vectorized(self, initial_text, context, sample_size=10, rollover=5):
        for w in self.posified_sentences_vectorized(initial_text, context, sample_size, rollover):
            yield w.split("::")[0]

    def posified_word_list(self, text, excluded_pos=[]):
        return [f"{w.text}::{w.pos_}" for w in self.nlp(text) if w.pos_ not in excluded_pos]

    def posified_sentences(self, initial_text, context):
        """
        Generator of posified words (word::pos) starting from `initial_text` 
        with a sliding window of `context` words.
        """

        ctx = self.posified_word_list(initial_text)
        if not ctx:
            return ""

        yield from ctx

        while True:
            next_words = self.graph.get(" ".join(ctx))
            if not next_words:
                # Try reducing the context window until we
                # find a node (or lose the entire context).
                ctx.pop(0)
                if not ctx:
                    return ""
                continue

            words = list(next_words.keys())
            weights = list(next_words.values())

            next_choice = random.choices(words, weights=weights, k=1)[0]

            ctx.append(next_choice)
            if len(ctx) > context:
                ctx.pop(0)

            yield next_choice

    def posified_sentences_vectorized(self, initial_text, context, sample_size=10, rollover=5):
        ctx = self.posified_word_list(initial_text)
        if not ctx:
            return ""

        yield from ctx

        current_text = initial_text
        current_vec = self.trained_model.get_text_vector(current_text)

        count = 0
        while True:
            next_words = self.graph.get(" ".join(ctx))
            if not next_words:
                # Try reducing the context window until we
                # find a node (or lose the entire context).
                ctx.pop(0)
                if not ctx:
                    return ""
                continue

            words = list(next_words.keys())

            next_choice = words[0]
            current_similarity = 0

            for word in random.sample(words, min(sample_size, len(words))):
                w = word.split("::")[0]

                next_text = f"{current_text} {w}"
                next_vector = self.trained_model.get_text_vector(next_text)

                similarity = (
                    np.dot(current_vec, next_vector) / 
                    (np.linalg.norm(current_vec) * np.linalg.norm(next_vector))
                )

                if similarity > current_similarity:
                    current_similarity = similarity
                    next_choice = word

            current_text = f"{current_text} {next_choice.split('::')[0]}"

            count += 1

            if count >= rollover:
                count = 0
                current_vec = self.trained_model.get_text_vector(current_text)

            ctx.append(next_choice)
            if len(ctx) > context:
                ctx.pop(0)

            yield next_choice

    def get_node(self, text):
        ctx = self.posified_word_list(text)
        if not ctx:
            return ""
        return self.graph.get(" ".join(ctx))


    def add_text(self, text, context):
        context = max(1, context)

        words = self.posified_word_list(
            text,
            excluded_pos=[
                "SPACE",
            ],
        )

        iter_parts = [words]
        for i in range(1, min(context, int(len(words) / 2))):
            iter_parts.append(words[i:])

        for group in zip(*iter_parts):
            m = group[0]
            for i in group[1:]:
                self.graph.setdefault(m, {}).setdefault(i, 0)
                self.graph[m][i] += 1

                m = f"{m} {i}"

        self.max_context = max(self.max_context, len(iter_parts))

