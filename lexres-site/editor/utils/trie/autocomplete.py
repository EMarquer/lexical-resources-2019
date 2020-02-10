try:
    from .structure  import Trie, TrieNode
except ImportError:
    from structure  import Trie, TrieNode
import typing as t
from sortedcontainers import SortedList

class TrieAutocompleter:
    trie: Trie

    def __init__(self, trie: Trie):
        self.trie = trie

    def get_node(self, word: str) -> t.Optional[TrieNode]:
        """Try to find the word in the trie and return the last node of the word, regardless of if the word ends up in
        a finishing node."""
        current_node = self.trie.root
        current_word = word
        while True:
            # if we emptied the word, it means we found it and we know the node
            if len(current_word) == 0:
                return current_node

            # if the word is not empty, we try to find the next character
            elif current_word[0] in current_node.children.keys():
                current_node = current_node.children[current_word[0]]
                current_word = current_word[1:]

            # if the word is not empty, and we did not find the next character, the word is not present in the trie
            else:
                return None

    def get_most_likely(self, word: str, start_node: TrieNode, max_selected: int=0, include_frequency: bool=False) -> (
            t.Union[t.List[str], t.List[t.Tuple[int, str]]]):
        """Explore the node to get the top most likely words"""
        if start_node is None:
            return []

        # sorted list of candidates (frequency, word up to node, node)
        candidate_nodes: t.List[t.Tuple[int, str, TrieNode]]
        candidate_nodes = SortedList([(start_node.frequency, word, start_node)], key = lambda x: x[0])

        # selected words, a list of words or a list of pairs (frequency, word) if include_frequency
        # sorted by decreasing frequency by construction
        selected_words: t.Union[t.List[str], t.List[t.Tuple[int, str]]] = []

        # continue to explore until we have no candidate nodes or enough selected words (if max_selected > 0)
        while len(candidate_nodes) > 0 and (max_selected <= 0 or len(selected_words) < max_selected):
            # grab the next most frequent node (last of the candidate list)
            frequency: int
            word_up_to_node: str
            candidate_node: TrieNode
            frequency, word_up_to_node, candidate_node = candidate_nodes.pop(-1)

            # if the node is an end word, add it to the selected words
            if candidate_node.is_leaf:
                selected_words.append((frequency, word_up_to_node) if include_frequency else word_up_to_node)

            # add all the children (if any) to the list of candidate
            for child in candidate_node.children.values():
                # insert the new candidate node (frequency, word up to node, node)
                candidate_nodes.add((child.frequency, word_up_to_node + child.value, child) )

        return selected_words

    def complete(self, word: str, max_selected: int=0, include_frequency: bool=False) -> t.List[str]:
        # return the input word if it is not found
        # find the node we currently have
        node = self.get_node(word)

        # find all the possible words from a specific node
        candidates = self.get_most_likely(word, node, max_selected=max_selected, include_frequency=include_frequency)

        return candidates

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

    completer = TrieAutocompleter(trie)
    print(completer.complete("wr"))
    print(completer.complete("wr", include_frequency=True))
    print(completer.complete("w"))
    print(completer.complete("w", include_frequency=True))
    print(completer.complete("", max_selected=5))
    print(completer.complete("", max_selected=5, include_frequency=True))
    print(completer.complete("Kebap"))
    print(completer.complete("Kebap", include_frequency=True))


    print("\n--- Lipsum corpus ---")
    with open("../experiments/lipsum.txt", "r", encoding="utf8") as f:
        text_lipsum = f.read().replace("\n", " ")
    trie_lipsum = Trie()
    trie_lipsum.train_on_text(text_lipsum)
    completer_lipsum = TrieAutocompleter(trie_lipsum)

    print(completer_lipsum.complete("lo"))
    print(completer_lipsum.complete("lo", include_frequency=True))
    print(completer_lipsum.complete("li"))
    print(completer_lipsum.complete("li", include_frequency=True))
    print(completer_lipsum.complete("l"))
    print(completer_lipsum.complete("l", include_frequency=True))
    print(completer_lipsum.complete("", max_selected=5))
    print(completer_lipsum.complete("", max_selected=5, include_frequency=True))
    print(completer_lipsum.complete("Kebap"))
    print(completer_lipsum.complete("Kebap", include_frequency=True))
