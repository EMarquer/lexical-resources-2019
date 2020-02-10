from .structure import Trie

def train_models():
    # load the data
    from . import tokenizer
    moliere_files = [os.path.join(THIS_FILE_DIRECTORY, "../../../experiments", f"molière_{i}.txt") for i in range(1,5)]
    corneille_files = [os.path.join(THIS_FILE_DIRECTORY, "../../../experiments", f"corneille_{i}.txt") for i in range(1,8)]

    def read_files(file_pathes):
        data = []
        for path in file_pathes:
            with open(path, 'r', encoding="utf8") as f:
                sentences = [sentence.strip() for sentence in tokenizer.sent_tokenize(f.read()) if sentence]
                data += sentences
        return data

    print(moliere_files)
    moliere_data = read_files(moliere_files)

    print(corneille_files)
    corneille_data = read_files(corneille_files)

    data = {
        'full': (moliere_data + corneille_data, {
            'moliere': moliere_data,
            'corneille': corneille_data
        })
    }

    print("Training the tries")
    train_tries(data)

def train_rnns(data, backoff_mechanism, backoff_model_name=None, encoder=None):
    if encoder is None: 
        encoder = CharEncoder()
        for model_name, (model_data, sub_models_data) in data:
            encoder.train(model_data)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #DEVICE = "cpu"

    n_epoch = 10
    batch_size = 128
    learning_rate = 1e-3
    collate = lambda sample: torch.tensor(
        encoder.add_padding([encoder.encode(sentence) for sentence in sample]),
        dtype=torch.long, device=DEVICE) # encode, pad and transform in a tensor

    from torch.utils.data import DataLoader, Dataset
    for model_name, (model_data, sub_models_data) in data:
        print(f"Training the rnn for '{model_name}'", end = ' ... ')
        dataset = Data(model_data) # larger dataset of the full data
        # train the rnns

def train_tries(data):
    for model_name, (model_data, sub_models_data) in data:
        print(f"Training the trie for '{model_name}'", end = ' ... ')
        trie = Trie()
        trie.train_on_text(" ".join(model_data))
        trie.save(os.path.join(THIS_FILE_DIRECTORY, TRIE_FOLDER, f"{model_name}.pkl"))
        print("Done")

        train_tries(sub_models_data)
# todo: add a condition before trying to run the training
#train_models()