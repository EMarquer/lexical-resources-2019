from .trie.structure import Trie
from .language_model.rnn_language_model import LanguageModel, BackoffMechanism, CharEncoder
from . import models
from . import tokenizer
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch
import torch.nn as nn
import os


DATA_DIRECTORY = os.path.join(models.THIS_FILE_DIRECTORY, "../../../dataset")
BOOK_DIRECTORY = os.path.join(DATA_DIRECTORY, "books")
AUTHOR_BOOKS_FILE = os.path.join(DATA_DIRECTORY, "movements_author_books.json")
LM_DIRECTORY = os.path.join(models.THIS_FILE_DIRECTORY, models.LM_FOLDER)
LM_BACKOFF_FILE = models.LM_BACKOFF_FILE
LM_ENCODER_FILE = models.LM_ENCODER_FILE
TRIE_FOLDER = os.path.join(models.THIS_FILE_DIRECTORY, models.TRIE_FOLDER)
MODEL_LIST_FILE = os.path.join(models.THIS_FILE_DIRECTORY, models.MODEL_LIST_FILE)

RNN_PARAMETERS = {
    "embedding_size": 16,
    "gru_hidden_size": 128,
    "gru_num_layers": 2,
    "gru_dropout": 0.2
}

RNN_TRAINING_PARAMETERS = {
    "n_epoch": 10,
    "batch_size": 128,
    "learning_rate": 1e-3
}

def get_data_files():
    """Returns a nested dictionary structure of the models and their files.
    
    A model on any level uses all the files listed in its sub-models."""
    import json
    with open(AUTHOR_BOOKS_FILE, 'r', encoding='utf8') as f:
        movement_author_book_dict = json.load(f)

    data = {
        models.DEFAULT_MODEL: {
            movement: {
                author: [os.path.join(BOOK_DIRECTORY, book + '.txt') for book in books] # create the list of book files
                for author, books in authors.items()
            } for movement, authors in movement_author_book_dict.items()
        }
    }

    return data

class FileDataset(Dataset):
    """A loader for the individual datafiles of an author"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.length = None
        self.sentences = None

    def __getitem__(self, index):
        return self.get_sentences()[index]

    def __len__(self):
        if self.length is None:
            self.length = len(self.get_sentences())

        return self.length

    def get_sentences(self):
        if self.sentences is None:
            with open(self.file_path, 'r', encoding="utf8") as f:
                self.sentences = [sentence.strip()
                    for sentence in tokenizer.sent_tokenize(f.read())
                    if sentence.strip()]
        return self.sentences

class AuthorDataset(ConcatDataset):
    def __init__(self, file_pathes):
        super().__init__([FileDataset(path) for path in file_pathes])
    
def get_datasets_recurse(data_files: dict, fallback_model_name=None):
    """Recursively creates a nested dictionary of model names as keys, dataset and sub-models as values."""

    output = dict()
    for model_name, sub_models_or_files in data_files.items():
        model_full_name = f"{fallback_model_name} -- {model_name}" if fallback_model_name is not None else model_name

        # when we arrive at the deepest reaches of the structure, whe have to create an AuthorDataset
        if isinstance(sub_models_or_files, list):
            output[model_full_name] = AuthorDataset(sub_models_or_files), None

        else:
            sub_models = get_datasets_recurse(sub_models_or_files, model_full_name)

            # the current model dataset is composed of the concatenation of only the first layer of sub-models dataset
            # (as those already contain the deeper layers)
            model_dataset = ConcatDataset([dataset for dataset, deeper_level in sub_models.values()])

            output[model_full_name] = model_dataset, sub_models

    return output

def get_datasets(data_files: dict):
    # flatten the datasets
    def flatten(model_data, fallback_model_name=None):
        flat = dict()
        for model_name, (dataset, sub_models) in model_data.items():
            # add the model
            flat[model_name] = dataset, fallback_model_name

            #add all the submodels
            if sub_models is not None:
                flat.update(flatten(sub_models, model_name))

        return flat

    return flatten(get_datasets_recurse(data_files))

def train_models():
    # get the datasets and fallback models for every model as a dict: {model_name: Dataset, fallback_model_name or None}
    datasets_dict = get_datasets(get_data_files())

    # create the model directories
    if not os.path.isdir(LM_DIRECTORY): os.mkdir(LM_DIRECTORY)
    if not os.path.isdir(TRIE_FOLDER): os.mkdir(TRIE_FOLDER)

    # train the models
    print("=== Training the language models ===")
    train_rnns(datasets_dict)
    print("=== Training the tries ===")
    train_tries(datasets_dict)
    print("=== Saving the model list ===")
    with open(MODEL_LIST_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join([model_name for model_name in datasets_dict.keys()]))

def train_rnns(data):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # prepare the encoder
    encoder = CharEncoder()
    for top_level_dataset, _ in data.values():
        encoder.train(top_level_dataset)

    # training hyperparameters
    n_epoch = RNN_TRAINING_PARAMETERS['n_epoch']
    batch_size = RNN_TRAINING_PARAMETERS['batch_size']
    learning_rate = RNN_TRAINING_PARAMETERS['learning_rate']

    # collate functions, applying the encoding and the padding for a batch
    collate_fn = lambda sample: torch.tensor(
        encoder.add_padding([encoder.encode(sentence) for sentence in sample]),
        dtype=torch.long, device=DEVICE) # encode, pad and transform in a tensor

    # backoff mechanism
    backoff_lm = BackoffMechanism()

    def train_rnn(dataset):
        """Train a single rnn on the data for a set number of epochs (no early stopping)"""

        # dataloader
        dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=True, batch_size=batch_size)

        # create model
        model = LanguageModel(len(encoder), **RNN_PARAMETERS)
        model = model.to(DEVICE)

        # create loss and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # training loop
        model.train()
        for epoch in range(n_epoch):
            running_loss, samples = 0, 0
            for sample in dataloader:
                # forward
                output, hidden = model(sample[:,:-1])
                loss = loss_function(output.transpose(-1, -2), sample[:,1:]) # train to predict next character
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # add to the epoch loss
                running_loss += loss.cpu().item()
                samples += 1
            
            # print the status for the epoch
            print(f"Epoch {epoch+1}/{n_epoch}: loss = {running_loss/samples}")

        return model.eval()

    for model_name, (model_dataset, fallback_model_name) in data.items():
        print(f"--- Training the rnn for '{model_name}' ---")
        model = train_rnn(model_dataset)

        backoff_lm.add(model, model_name, fallback_model_name)
        # train the rnns
    
    # save everything
    backoff_lm.save(LM_DIRECTORY, LM_BACKOFF_FILE)
    encoder.save(os.path.join(LM_DIRECTORY, LM_ENCODER_FILE))

def train_tries(data):
    for model_name, (model_data, fallback_model_name) in data.items():
        print(f"Training the trie for '{model_name}'", end = ' ... ')
        trie = Trie()
        trie.train_on_text(" ".join(model_data))
        trie.save(os.path.join(TRIE_FOLDER, f"{model_name}.pkl"))
        print("Done")
