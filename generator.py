import sys
import time
import random
import rapidjson as json
import spacy


def posified_word_list(doc, excluded_pos=[]):
    return [f"{w.text}::{w.pos_}" for w in doc if w.pos_ not in excluded_pos]


class Generator:
    def __init__(self, graph, nlp, context):
        assert context >= 1

        self.graph = graph
        self.nlp = nlp
        self.context = context

    def sentences(self, initial_text):
        for w in self.posified_sentences(initial_text):
            yield w.split("::")[0]

    def posified_sentences(self, initial_text):
        ctx = posified_word_list(self.nlp(initial_text))
        if not ctx:
            return ""

        yield from ctx

        while True:
            next_words = self.graph.get(" ".join(ctx))
            if not next_words:
                ctx.pop(0)
                if not ctx:
                    return ""
                continue

            next_choice = random.choices(
                list(next_words.keys()), weights=list(next_words.values()), k=1
            )[0]

            ctx.append(next_choice)
            if len(ctx) > self.context:
                ctx.pop(0)

            yield next_choice

    def get_node(self, text):
        ctx = posified_word_list(self.nlp(text))
        if not ctx:
            return ""
        return self.graph.get(" ".join(ctx))

    def corpus_parser(self, text):
        context = max(1, context)

        words = posified_word_list(
            nlp(text),
            excluded_pos=[
                "SPACE",
            ],
        )

        iter_parts = [words]
        for i in range(1, min(context + 1, int(len(words) / 2))):
            iter_parts.append(words[i:])

        for group in zip(*iter_parts):
            m = group[0]
            for i in group[1:]:
                graph.setdefault(m, {}).setdefault(i, 0)
                graph[m][i] += 1

                m = f"{m} {i}"



