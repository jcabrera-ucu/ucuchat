import sys
import time
import random
import rapidjson as json
import spacy

from cmd import Cmd


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


def corpus_parser(graph, nlp, text, context):
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


class UcuChat(Cmd):
    prompt = ">> "

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nlp = spacy.load("es_core_news_sm")

        self.graph = {}

        self.min_word_count = 50
        self.context = 1

    def do_node(self, arg):
        generator = Generator(graph=self.graph, nlp=self.nlp, context=self.context)
        print(generator.get_node(arg))

    def do_load(self, arg):
        """Load graph from json file (clears the currently loaded graph from memory)"""

        try:
            start_time = time.time()
            with open(arg) as f:
                self.graph = json.loads(f.read())
            print(f"OK ({(time.time() - start_time):.1f} s)")
        except Exception as e:
            print(e)

    def do_clear(self, arg):
        """Clears the currently loaded graph from memory"""

        self.graph = {}

    def help_add(self):
        print("Expands the in-memory graph with the contents of a corpus file")
        print()
        print("Usage:")
        print("  >> add <number> path/to/corpus/file.txt")
        print("     Where <number> is the size of the context window to use")

    def do_add(self, arg):
        try:
            start_time = time.time()

            ctx = int(arg.split(" ")[0])
            path = " ".join(arg.split(" ")[1:])

            with open(path) as f:
                for text in f.readlines():
                    corpus_parser(
                        text=text, graph=self.graph, nlp=self.nlp, context=ctx
                    )
            print(f"OK ({(time.time() - start_time):.1f} s)")
        except Exception as e:
            print(e)

    def do_save(self, arg):
        """Saves the in-memory graph to a json file

        Usage:
            >> save path/to/file.json
        """

        try:
            open(arg, "w").write(json.dumps(self.graph))
        except Exception as e:
            print(e)

    def do_info(self, arg):
        print(f"graph:node_count = {len(self.graph)}")
        print(f"min_word_count   = {self.min_word_count}")
        print(f"context          = {self.context}")

    def do_words(self, arg):
        try:
            self.min_word_count = int(arg)
        except Exception as e:
            print(e)

    def do_context(self, arg):
        try:
            self.context = int(arg)
        except Exception as e:
            print(e)

    def default(self, inp):
        generator = Generator(graph=self.graph, nlp=self.nlp, context=self.context)

        count = 0
        for w in generator.sentences(inp):
            count += 1
            sys.stdout.write(f"{w} ")
            sys.stdout.flush()

            if (
                count >= self.min_word_count
                and w == "."
                or count >= self.min_word_count * 2
            ):
                break
        print()


UcuChat().cmdloop()
