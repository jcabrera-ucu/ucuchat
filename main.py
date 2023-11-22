import os

import torch.nn as nn
from model.data_loader import *
from model.utils import *
from model.word2vec import Word2VecModel
from model.training import Trainer

config = {
    'model_to_use': 'skipgram',
    'model_dir': 'ucuchat_model/wikipedia_es/skipgram',
    'data_dir': 'wikipedia-es(trained_model)/',
    'epochs': 5,
    'learning_rate': 0.025,
    'checkpoint_frequency': None,
    'optimizer': 'Adam',
    'shuffle': True,
    'train_batch_size': 8,
    'train_steps': None,
    'val_batch_size': 8,
    'val_steps': None
}


def train():
    os.makedirs(config['model_dir'])
    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name=config['model_to_use'],
        data_dir=config['data_dir'],
        batch_size=config['train_batch_size'],
        shuffle=config['shuffle'],
        vocab=None
    )
    val_dataloader, _ = get_dataloader_and_vocab(
        model_name=config['model_to_use'],
        data_dir=config['data_dir'],
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
    lr_scheduler = get_lr_scheduler(optimizer, config['epochs'], verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
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
    torch.cuda.empty_cache()
    trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config['model_dir'])
    save_config(config, config['model_dir'])
    print(f"Model artifacts saved to folder: {config['model_dir']}")


if __name__ == '__main__':
    if not os.path.exists(config['model_dir']):
        train()
    else:
        trained_model = Word2VecModel(config['model_dir'])
        word = 'semana'
        similar_words = trained_model.get_similar_words(word, 20)
        if similar_words is not None:
            for word, sim in similar_words.items():
                print("{}: {:.3f}".format(word, sim))
