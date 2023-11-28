import sys
import time
import random
import json
import spacy

from glob import glob

from cmd import Cmd
from graph import Graph
from model.word2vec import Word2VecModel


class UcuChat(Cmd):
    prompt = ">> "

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nlp = spacy.load("es_core_news_sm")
        self.min_word_count = 50
        self.context = 4
        self.vectorized_sample_size = 10
        self.vectorized_rollover = 5
        self.trained_model = Word2VecModel('ucuchat_model/wikipedia_es/skipgram')
        self.graph = Graph(self.nlp, self.trained_model)

    def do_sim(self, arg):
        similar_words = self.trained_model.get_similar_words(arg, 20)
        if similar_words is not None:
            for word, sim in similar_words.items():
                print("{}: {:.3f}".format(word, sim))

    def do_node(self, arg):
        print(self.graph.get_node(arg))

    def do_load(self, arg):
        """Load graph from json file (clears the currently loaded graph from memory)"""

        try:
            start_time = time.time()
            with open(arg) as f:
                self.graph.load(json.loads(f.read()))
            print(f"OK ({(time.time() - start_time):.1f} s)")
        except Exception as e:
            print(e)

    def do_clear(self, arg):
        """Clears the currently loaded graph from memory"""

        self.graph = Graph(self.nlp, self.trained_model)

    def do_add(self, arg):
        """Expands the in-memory graph with the contents of a corpus file

        Usage:
          >> add <number> path/to/corpus/file.txt
          >> add <number>,<sample-size> path/to/corpus/file.txt

             Where <number> is the size of the context window to use
             and <sample-size> is the maximum number of files to sample

        Examples:
          >> add 4 path/to/corpus/file.txt
          >> add 4 path/to/**/*.txt
          >> add 4,100 path/to/**/*.txt
        """

        try:
            start_time = time.time()

            num_part = arg.split(" ")[0].split(",")
            ctx = int(num_part[0])
            path = " ".join(arg.split(" ")[1:])
            
            file_list = glob(path)
            if len(num_part) > 1:
                file_list = random.sample(file_list, int(num_part[1]))

            for file_path in file_list:
                with open(file_path, encoding="utf-8") as f:
                    sys.stdout.write(f"Reading: {file_path}")
                    sys.stdout.flush()

                    for text in f.readlines():
                        self.graph.add_text(text, context=ctx)

                    sys.stdout.write(f" OK\n")

            print(f"OK ({(time.time() - start_time):.1f} s)")
        except Exception as e:
            print(e)

    def do_save(self, arg):
        """Saves the in-memory graph to a json file

        Usage:
            >> save path/to/file.json
        """

        try:
            open(arg, "w").write(json.dumps(self.graph.as_dict()))
        except Exception as e:
            print(e)

    def do_info(self, arg):
        print(f"graph:node_count       = {len(self.graph.graph):_}")
        print(f"graph:max_context      = {self.graph.max_context}")
        print(f"vectorized:sample_size = {self.vectorized_sample_size}")
        print(f"vectorized:rollover    = {self.vectorized_rollover}")
        print(f"min_word_count         = {self.min_word_count}")
        print(f"context                = {self.context}")

    def do_words(self, arg):
        try:
            self.min_word_count = int(arg)
        except Exception as e:
            print(e)

    def do_sample_size(self, arg):
        try:
            self.vectorized_sample_size = int(arg)
        except Exception as e:
            print(e)

    def do_rollover(self, arg):
        try:
            self.vectorized_rollover = int(arg)
        except Exception as e:
            print(e)

    def do_context(self, arg):
        try:
            self.context = int(arg)
        except Exception as e:
            print(e)

    def do_shell(self, arg):
        try:
            from IPython import embed
            embed(colors='neutral')
        except Exception as e:
            print(e)

    def default(self, inp):
        text = ""
        count = 0
        for w in self.graph.sentences(inp, self.context):
            count += 1
            text += f"{w} "
            sys.stdout.write(f"{w} ")
            sys.stdout.flush()

            if (
                count >= self.min_word_count
                and w == "."
                or count >= self.min_word_count * 2
            ):
                break
        print()

    def do_vectorized(self, inp):
        text = ""
        count = 0
        for w in self.graph.sentences_vectorized(
                inp, 
                self.context, 
                sample_size=self.vectorized_sample_size, 
                rollover=self.vectorized_rollover):
            count += 1
            text += f"{w} "
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
