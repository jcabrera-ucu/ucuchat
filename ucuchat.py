import sys
import time
import random
import json
import spacy

from glob import glob

from cmd import Cmd



# ==================================================
# ==================================================
import os

import torch.nn as nn
from model.data_loader import *
from model.utils import *
from model.word2vec import Word2VecModel
from model.training import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau

config = {
    'model_to_use': 'skipgram',
    'model_dir': 'ucuchat_model/wikipedia_es/skipgram',
    'data_dir': 'wikipedia-es/',
    'data_partitions': 3,
    'epochs': 5,
    'learning_rate': 0.01,
    'checkpoint_frequency': None,
    'optimizer': 'Adam',
    'shuffle': True,
    'train_batch_size': 4,
    'train_steps': None,
    'val_batch_size': 4,
    'val_steps': None
}


def train():
    os.makedirs(config['model_dir'])
    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name=config['model_to_use'],
        data_dir=config['data_dir'],
        data_partitions=config['data_partitions'],
        batch_size=config['train_batch_size'],
        shuffle=config['shuffle'],
        vocab=None
    )
    val_dataloader, _ = get_dataloader_and_vocab(
        model_name=config['model_to_use'],
        data_dir=config['data_dir'],
        data_partitions=config['data_partitions'],
        batch_size=config['val_batch_size'],
        shuffle=config['shuffle'],
        vocab=vocab
    )

    vocab_size = len(vocab.get_stoi())
    print(f'Vocabulary size: {vocab_size}')

    model_class = get_model_class(config['model_to_use'])
    model = model_class(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()

    optimizer_class = get_optimizer_class(config['optimizer'])
    optimizer = optimizer_class(model.parameters(), lr=config['learning_rate'])
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        data_partition=config['data_partitions'],
        epochs=config['epochs'],
        train_dataloader=train_dataloader,
        train_steps=config['train_steps'],
        val_dataloader=val_dataloader,
        val_steps=config['val_steps'],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config['checkpoint_frequency'],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config['model_dir'],
        model_name=config['model_to_use'],
    )
    trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config['model_dir'])
    save_config(config, config['model_dir'])
    print(f"Model artifacts saved to folder: {config['model_dir']}")


# if __name__ == '__main__':
#     if not os.path.exists(config['model_dir']):
#         train()
#     else:
#         trained_model = Word2VecModel(config['model_dir'])
#         word = 'francia'
#         similar_words = trained_model.get_similar_words(word, 20)
#         if similar_words is not None:
#             for word, sim in similar_words.items():
#                 print("{}: {:.3f}".format(word, sim))
# 

# ==================================================
# ==================================================


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

        self.trained_model = Word2VecModel(config['model_dir'])

    def do_sim(self, arg):
        similar_words = self.trained_model.get_similar_words(arg, 20)
        if similar_words is not None:
            for word, sim in similar_words.items():
                print("{}: {:.3f}".format(word, sim))

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

    # def help_add(self):
    #     print("Expands the in-memory graph with the contents of a corpus file")
    #     print()
    #     print("Usage:")
    #     print("  >> add <number> path/to/corpus/file.txt")
    #     print("     Where <number> is the size of the context window to use")
    #     print()
    #     print("Examples:")
    #     print("  >> add 4 path/to/corpus/file.txt")
    #     print("  >> add 4 path/to/**/*.txt")

    def do_add(self, arg):
        """Expands the in-memory graph with the contents of a corpus file

        Usage:
          >> add <number> path/to/corpus/file.txt
             Where <number> is the size of the context window to use

        Examples:
          >> add 4 path/to/corpus/file.txt
          >> add 4 path/to/**/*.txt
        """

        try:
            start_time = time.time()

            ctx = int(arg.split(" ")[0])
            path = " ".join(arg.split(" ")[1:])

            for file_path in glob(path):
                with open(file_path, encoding="utf-8") as f:
                    sys.stdout.write(f"Reading: {file_path}")
                    sys.stdout.flush()

                    for text in f.readlines():
                        corpus_parser(
                            text=text, graph=self.graph, nlp=self.nlp, context=ctx
                        )

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
            open(arg, "w").write(json.dumps(self.graph))
        except Exception as e:
            print(e)

    def do_info(self, arg):
        print(f"graph:node_count = {len(self.graph):_}")
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
