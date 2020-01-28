from __future__ import annotations
import typing as t
from pprint import pprint

class TrieNode:
    """A trie node.

    The root should have None as its value, and leafs should have `is_leaf` True.
    """
    children: t.Dict[str, TrieNode]
    is_leaf: bool
    frequency: int
    value: str

    def __init__(self, value: str, frequency: int=1, is_leaf: bool=False, children: t.Dict[str, TrieNode]=None):
        self.children = {} if children is None else children
        self.value = value
        self.frequency = frequency
        self.is_leaf = is_leaf
    
    def add_word(self, word: str):
        if word == "":
            pass

        else:
            # handle current character
            if word[0] not in self.children.keys():
                self.children[word[0]] = TrieNode(word[0])
            else:
                self.children[word[0]].frequency += 1

            # handle potential next characters
            if len(word) == 1:
                self.children[word[0]].is_leaf = True
            else:
                self.children[word[0]].add_word(word[1:])

    def get_frequency_dict(self) -> t.Tuple[t.Tuple[str, int, bool], t.Dict]:
        children_dicts = [child.get_frequency_dict() for child in self.children.values()]

        return (self.value, self.frequency, self.is_leaf), {key: value for key, value in children_dicts}
    
    def contains(self, word: str) -> bool:
        if word == "": return False
        elif len(word) == 1: return self.is_leaf and self.value == word
        else: return self.value == word[0] and any(child.contains(word[1:]) for child in self.children.values())

class Trie:
    root: TrieNode

    def __init__(self):
        self.root = TrieNode(None, frequency=0)

    def add_word(self, word: str):
        self.root.add_word(word)

    def get_frequency_dict(self) -> t.Tuple[t.Tuple[str, int, bool], t.Dict]:
        return self.root.get_frequency_dict()

    def train_on_text(self, text: str):
        import re
        tokens = re.findall(r"([\w-]+)", text.lower())
        for token in tokens:
            self.add_word(token)

    def contains(self, word: str) -> bool:
        return any(child.contains(word) for child in self.root.children.values())

    

if __name__ == "__main__":
    trie = Trie()
    text = ("A very large number of published documents contain text only. They" +
    "often look boring, and they are often written in obscure language, using " +
    "mile-long sentences and cryptic technical terms, using one font only, " +
    "perhaps even without headings. Such style, or lack of style, might be the" +
    " one you are strongly expected to follow when writing eg scientific or " +
    "technical reports, legal documents, or administrative papers. It is " +
    "natural to think that such documents would benefit from a few " +
    "illustrative images. (However, just adding illustration might be rather" +
    " useless, if the text remains obscure and unstructured.)")

    trie.train_on_text(text)

    

    pprint(trie.get_frequency_dict(), width=120)

    print("'a'", trie.contains("a"))
    print("''", trie.contains(""))
    print("'reports'", trie.contains("reports"))
    print("*'reportss'", trie.contains("reportss"))

#print(TrieNode("a", [TrieNode("b")]).is_leaf)