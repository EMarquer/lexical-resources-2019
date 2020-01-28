import torch
import torch.nn as nn
import typing as t
from os.path import join as join_path
import pickle
try:
    from .char_encoder import CharEncoder
except:
    from char_encoder import CharEncoder

class LanguageModel(nn.Module):
    def __init__(self,
            vocabulary_size: int,
            embedding_size: int,
            gru_hidden_size: int,
            gru_num_layers: int,
            gru_dropout: float=0):
        super().__init__()

        # store the values used when construct the model to be able to re-build it when loading
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.gru_dropout = gru_dropout

        # embedding layer
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

        # recurrent layer
        self.gru = nn.GRU(embedding_size, gru_hidden_size, gru_num_layers, batch_first=True, dropout=gru_dropout)
        
        # output layer
        self.output = nn.Linear(gru_hidden_size, vocabulary_size)

    def forward(self, input: torch.LongTensor, hidden: t.Optional[torch.Tensor]=None):
        out = self.embedding(input)
        out, hidden = self.gru(out) if hidden is None else self.gru(out, hidden)
        return self.output(out), hidden

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            torch.save({
                # store the construction parameters to be able to rebuild the model as is
                'construction_parameters': {
                    'vocabulary_size': self.vocabulary_size,
                    'embedding_size': self.embedding_size,
                    'gru_hidden_size': self.gru_hidden_size,
                    'gru_num_layers': self.gru_num_layers,
                    'gru_dropout': self.gru_dropout 
                },
                'state_dict': self.state_dict()}, f)

    @classmethod
    def load(cls, file_path: str, device: str) -> 'LanguageModel':
        # load all the necessary data from the file
        with open(file_path, 'rb') as f:
            data = torch.load(f, map_location=device)
        
        # rebuild the model
        model = cls(**data['construction_parameters'])

        # load state
        model.load_state_dict(data['state_dict'])

        return model


class BackoffMechanism:
    # a dictionary with a key for the model, and a tuple of the model and the key of the other model to backoff to
    models: t.Dict[str, t.Tuple[LanguageModel, t.Optional[str]]]
    backoff_rate: float  # value between 0 and 1 of the weight of the backoff model in the interpolation process


    def __init__(self, backoff_rate: float=0.3):
        """Interpolation-based backoff"""
        self.models = dict()
        self.backoff_rate = backoff_rate
        self.log_softmax = nn.LogSoftmax(dim=-1)  # is used for sampling
        self.softmax = nn.Softmax(dim=-1)  # is used for sampling

    def add(self, model: LanguageModel, model_key: str, backoff_key: str=None):
        self.models[model_key] = (model.eval(), backoff_key)

    def run_model(self, model_key: str, input: torch.LongTensor, hiddens: t.Optional[t.List[torch.Tensor]]=None):
        model, backoff_key = self.models[model_key]

        if backoff_key is None or backoff_key not in self.models.keys():
            if hiddens is None or len(hiddens) < 1:
                model_output, model_hidden = model(input, None)
            else:
                model_output, model_hidden = model(input, hiddens[0])
            
            return self.log_softmax(model_output), [model_hidden]
        
        else:
            if hiddens is None or len(hiddens) < 1:
                model_output, model_hidden = model(input, None)
                backoff_output, backoff_hiddens = self.run_model(backoff_key, input, None)
            else:
                model_output, model_hidden = model(input, hiddens[0])
                backoff_output, backoff_hiddens = self.run_model(backoff_key, input, hiddens[1:])

            # apply interpolation
            interpolation_output = self.log_softmax(model_output) * (1 - self.backoff_rate) + backoff_output * self.backoff_rate

            # reapply softmax to be sure we have a probability distribution
            #interpolation_output = self.softmax(interpolation_output)
            
            return (interpolation_output, [model_hidden] + backoff_hiddens)  # sequence of all the hidden states
        
    def sample(self,
                model_key: str,
                char_encoder: CharEncoder,
                context: str="",
                word_count: int=1,
                strategy: str="max",
                device="cpu")-> t.List[str]:
        """
        strategy: either 'random' or 'max' depending on if we use a random sampling or take the max
        """
        token_boundary_id = char_encoder.char_to_id[char_encoder.token_boundary]
        characters: t.List[torch.LongTensor] = []
        produced_words = []
        hidden = None
        
        # if we have some context, first we run the rnn on this context
        if context:
            # encode the context and remove the last boundary character 
            # (to start the prediction at the same point as when there is no context)
            context_encoded = torch.LongTensor([char_encoder.encode(context)])[:,:-1].to(device)
            out, hidden = self.run_model(model_key, context_encoded, hidden)

        # if we have no context we start from a boundary character
        # if we do have a context, force the last character of the context to be a word boundary
        last_character = torch.LongTensor([[token_boundary_id]]).to(device)

        while len(produced_words) < word_count and len(characters) < 30:
            # compute 1 step of the model
            out, hidden = self.run_model(model_key, last_character, hidden)

            # get the index of the new character
            if strategy == "random":
                last_character = torch.multinomial(self.softmax(out[0]).cpu(), 1).to(device)
            else:
                last_character = torch.LongTensor([[out[0].cpu().argmax(-1)]]).to(device)
            
            # store the new character
            characters.append(last_character.cpu().item())

            # check if the new character is an end of word
            # if so, add the new word to the list and reset the list of stored characters
            if characters[-1] == token_boundary_id: 
                produced_words.append(char_encoder.decode(characters[:-1]))
                characters = []

        return produced_words

    def save(self, folder_path: str, backoff_file_name: str):
        # prepare a dictionary mapping each model name to a file in which said model will be saved and the name of the
        # corresponding backoff model
        file_and_backoff_dict = {
            name: (join_path(folder_path, name + ".tch"), backoff_name) for name, (lm, backoff_name) in self.models.items()
        }

        # save all the models in their respective files
        for model_name, (file_path, backoff_name) in file_and_backoff_dict.items():
            self.models[model_name][0].save(file_path)

        # save the file_and_backoff_dict in a specific file
        with open(join_path(folder_path, backoff_file_name), 'wb') as f:
            pickle.dump({'backoff_rate': self.backoff_rate, 'file_and_backoff_dict': file_and_backoff_dict}, f)

    @classmethod
    def load(cls, folder_path: str, backoff_file_name: str, device='cpu') -> 'BackoffMechanism':
        """backoff_file_name added to the folder_path and should contain the extension"""
        # load all the necessary data from the file
        with open(join_path(folder_path, backoff_file_name), 'rb') as f:
            data = pickle.load(f)
        backoff_rate = data['backoff_rate']

        # recreate the object
        backoff_mechanism = cls(backoff_rate)

        # load all the models and the links between the models
        for model_name, (file_path, backoff_name) in data['file_and_backoff_dict'].items():
            backoff_mechanism.add(LanguageModel.load(file_path, device), model_name, backoff_name)
        
        return backoff_mechanism

if __name__ == "__main__":
    with open('lipsum_sentences.txt', 'r', encoding="utf8") as f:
        data = [line.strip() for line in f]

    encoder = CharEncoder()
    encoder.train(data)

    from torch.utils.data import DataLoader, Dataset

    # small dataset from a list
    class Data(Dataset):
        def __init__(self, data):
            self.data = data
        def __getitem__(self, index):
            return self.data[index]
        def __len__(self):
            return len(self.data)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #DEVICE = "cpu"

    n_epoch = 100
    batch_size = 25
    learning_rate = 1e-3
    collate = lambda sample: torch.tensor(
        encoder.add_padding([encoder.encode(sentence) for sentence in sample]),
        dtype=torch.long, device=DEVICE) # encode, pad and transform in a tensor
    
    large_dataset = Data(data) # larger dataset of the full data
    large_dataloader = DataLoader(large_dataset, collate_fn=collate, shuffle=True, batch_size=batch_size)
    small_dataset = Data(data[:int(.3*len(data))]) # smaller dataset of 30% of the full data
    small_dataloader = DataLoader(small_dataset, collate_fn=collate, shuffle=True, batch_size=batch_size)

    large_lm = LanguageModel(len(encoder), 16, 128, 2, 0.2)
    small_lm = LanguageModel(len(encoder), 16, 128, 2, 0.2)

    backoff_lm = BackoffMechanism(0.1)
    backoff_lm.add(large_lm, "large")
    backoff_lm.add(small_lm, "small", "large")

    # train the two models
    for model, dataloader in [(large_lm.to(DEVICE), large_dataloader), (small_lm.to(DEVICE), small_dataloader)]:
        model.train()
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(n_epoch):
            running_loss, samples = 0, 0
            for sample in dataloader:
                optimizer.zero_grad()
                output, hidden = model(sample[:,:-1])
                loss = loss_function(output.transpose(-1, -2), sample[:,1:]) # train to predict next character
                loss.backward()
                optimizer.step()
                running_loss += loss.cpu().item()
                samples += 1
            print(f"Epoch {epoch+1}/{n_epoch}: loss = {running_loss/samples}")
        model.eval()

    # generate next word using backoff
    print(backoff_lm.sample("large", encoder, device=DEVICE))
    print(backoff_lm.sample("small", encoder, device=DEVICE))
    print(backoff_lm.sample("large", encoder, "Lorem", device=DEVICE))
    print(backoff_lm.sample("small", encoder, "Lorem", device=DEVICE))
    print(backoff_lm.sample("large", encoder, "Lorem", 2, device=DEVICE))
    print(backoff_lm.sample("small", encoder, "Lorem", 2, device=DEVICE))