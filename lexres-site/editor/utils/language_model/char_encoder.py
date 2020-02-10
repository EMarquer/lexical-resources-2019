import torch
import torch.nn as nn
import typing as t
import re
import pickle
from ..tokenizer import tokenize

class CharEncoder:
    char_to_id: t.Dict[str, int]
    id_to_char: t.List[str]
    token_boundary: str # token boundary character
    padding: str # padding character
    unknown: str # unknown character

    def __init__(self, token_boundary: str=" ", padding: str="_", unknown: str="?"):
        self.token_boundary = token_boundary
        self.padding = padding
        self.unknown = unknown

    def encode(self, sentence: str) -> t.List[int]:
        tokens = tokenize(sentence)

        # encode all the characters and add token boundaries at the extremities of the sentence and between each token
        encoded_tokens = [
            ([self.char_to_id[self.token_boundary]] + # add boundary at the begining of each token
             [self.char_to_id.get(char, self.char_to_id[self.unknown]) for char in token]) # encode token
            for token in tokens] + [[self.char_to_id[self.token_boundary]]] # add boundary at the end of the sequence

        # transform into a flat list
        encoded_chars = [encoded_char for encoded_token in encoded_tokens for encoded_char in encoded_token]

        return encoded_chars

    def decode(self, sentence: t.List[int]) -> str:
        return ''.join(self.id_to_char[char_id] for char_id in sentence)

    def train(self, sentences: t.List[str]):
        char_set = set()

        # gather all the characters
        for sentence in sentences:
            tokens = tokenize(sentence)
            for token in tokens:
                char_set.update(token)

        # create the id to character list
        self.id_to_char = [self.token_boundary, self.padding, self.unknown] + list(char_set)

        # create the character to id dict
        self.char_to_id = {char: char_id for char_id, char in enumerate(self.id_to_char)}
        
    def add_padding(self, sentences: t.List[t.List[int]]) -> t.List[t.List[int]]:
        max_length = max(map(len, sentences))

        # pad the sentences until the max length
        padded_sentences = [
            sentence + ([self.char_to_id[self.padding]] * (max_length - len(sentence)))
            for sentence in sentences]

        return padded_sentences

    def save(self, file_name: str):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name: str) -> "CharEncoder":
        with open(file_name, 'rb') as f:
            encoder = pickle.load(f)
        return encoder

    def __len__(self):
        return len(self.id_to_char)

if __name__ == "__main__":
    data = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Vulputate mi sit amet mauris.",
        "Scelerisque fermentum dui faucibus in ornare quam.",
        "Nisi quis eleifend quam adipiscing.",
        "Et molestie ac feugiat sed lectus vestibulum mattis.",
        "Dui nunc mattis enim ut tellus.",
        "Quis hendrerit dolor magna eget est.",
        "Mus mauris vitae ultricies leo integer.",
        "Arcu non sodales neque sodales ut etiam sit amet nisl.",
        "Ipsum dolor sit amet consectetur adipiscing elit pellentesque.",
        "Vestibulum mattis ullamcorper velit sed ullamcorper morbi tincidunt ornare.",
        "Eu augue ut lectus arcu bibendum at varius vel pharetra.",
        "Convallis aenean et tortor at risus viverra.",
        "Odio euismod lacinia at quis.",
        "Fermentum leo vel orci porta non pulvinar.",
        "Fermentum dui faucibus in ornare quam viverra.",
        "Pellentesque habitant morbi tristique senectus et netus et malesuada fames.",
        "Parturient montes nascetur ridiculus mus.",
        "Suspendisse in est ante in nibh.",
        "At augue eget arcu dictum varius.",
        "Est ante in nibh mauris cursus mattis.",
        "Morbi quis commodo odio aenean sed.",
        "Dignissim enim sit amet venenatis urna cursus eget.",
        "Amet porttitor eget dolor morbi non.",
        "Diam in arcu cursus euismod quis viverra nibh cras.",
        "Sit amet massa vitae tortor condimentum lacinia.",
        "Ac turpis egestas maecenas pharetra.",
        "Placerat orci nulla pellentesque dignissim enim sit.",
        "Amet volutpat consequat mauris nunc congue nisi vitae suscipit.",
        "Eget gravida cum sociis natoque penatibus et magnis dis."]
    
    encoder = CharEncoder(token_boundary='|')
    encoder.train(data)

    print("--- test encode ---")
    print(encoder.encode("I am a cat!?"))
    print("--- test decode ---")
    print(encoder.decode(encoder.encode("I am a cat!?")))

    print("--- test padding ---")
    print('\n'.join([encoder.decode(sentence) for sentence in 
        encoder.add_padding([encoder.encode(sentence) for sentence in data])]))