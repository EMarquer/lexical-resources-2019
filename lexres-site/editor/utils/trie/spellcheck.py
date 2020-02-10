try:
    from .structure import Trie, TrieNode
except ImportError:
    from structure import Trie, TrieNode
import typing as t
import itertools
from sortedcontainers import SortedList, SortedSet

def sortest_distance_to_next_leaf(start_node: TrieNode) -> t.Tuple[int, TrieNode, str]:
    """Get closest leaf and the corresponding path, using BFS."""
    current_level = {(0, start_node, "")}
    found = False

    while not found:
        next_level = set()
        for current_cost, current_node, current_word in current_level:
            # if the currently analysed node is a leaf, no need to go further
            if current_node.is_leaf:
                return current_cost, current_node, current_word
            
            # if the currently analysed node is not a leaf, add all its children (if any) to the next level to analyse
            else:
                for child in current_node.children.values():
                    next_level.add((current_cost + 1, child, current_word + child.value))

        current_level = next_level

    # filler value, should never be used
    return -1, ""
                

class TrieSpellchecker:
    trie: Trie

    def __init__(self, trie: Trie):
        self.trie = trie
    
    def check_correct(self, word: str) -> str:
        """"""
        if self.trie.contains(word): return word
        else:
            candidates = self.trie_levenshtein_spellcorrect(word)
            return candidates[0][1] # get the text part of the first candidate

    def trie_levenshtein_spellcorrect(self,
            word: str,
            number_of_results: int=1,
            max_distance: int=-1,
            verbose=False) -> (t.List[t.Tuple[int, str]]):
        """Explore the trie to find the closest existing words to the provided one using Levenshtein distance.
        
        Based on a de-recurred version Levenshtein edit distance, then adapted to exploring a trie.
        Non-recursive version to optimise search into breath-first instead of recursive depth-first.
        Also, the "correct word" is stored when going through the trie, and least costly candidate branches are 
        considered first.

        :param word: the word from which we want the closest in the trie
        :param number_of_results: (default 1) the number of words from the trie to output
        :param max_distance: (default -1) the maximal acceptable Levenshtein distance, ignored if 0 or lower
        :param verbose: (default False) if True, will print the reason to stop the algorithm, between:
        - we find enough valid final nodes;
        - we have no more candidates;
        - we have no more candidates under the cost threshold.

        :return: a list of (cost, word from the trie) tuples, with up to `number_of_results` elements
        """
        # store all the current states in a sorted set, this allows to work only with the less costly option
        # (one set is a tuple (cost, node, remaining word  characters, word up to node))
        candidate_cost_node_word_set: t.Set[t.Tuple[int, TrieNode, str, str]]
        candidate_cost_node_word_set = SortedSet(
            iterable={(0, self.trie.root, word, "")},
            # the key is the cost, then the frequency, then the remaining length 
            key=lambda x: (x[0], len(x[2]), x[1].frequency))

        # store the optimal distance result
        optimal_cost_node_word_list: t.Set[t.Tuple[int, str]]
        optimal_cost_node_word_list = SortedSet()

        # loop on all the nodes until either:
        while ( # we find enough valid final nodes
                len(optimal_cost_node_word_list) < number_of_results and  
                # we have no more candidates
                len(candidate_cost_node_word_set) > 0 and  
                # we have no more candidates under the cost threshold
                (max_distance < 0 or candidate_cost_node_word_set[0][0] < max_distance)):  

            # get the least costly candidate
            current_cost, current_node, current_word_left, current_correct_word = candidate_cost_node_word_set.pop(0)

            # --- stop condition ---
            # we consider a solution as valid if the current node is a leaf and the word to study is empty
            if len(current_word_left) == 0 and current_node.is_leaf:
                optimal_cost_node_word_list.add((current_cost, current_correct_word))
            
            # --- previous base cases ---
            # the word to study is empty or the branch is a cul-de-sac
            elif len(current_word_left) == 0: # the word is empty
                # get remaining distance to the next leaf and the corresponding node and word
                # (number of additions to go from the current word to a true word)
                cost_to_next_leaf, next_leaf, word_to_next_leaf = sortest_distance_to_next_leaf(current_node)
                candidate_cost_node_word_set.add(
                    (current_cost + cost_to_next_leaf, next_leaf, "", current_correct_word + word_to_next_leaf))

            elif len(current_node.children) < 1: # the branch is a cul-de-sac
                # if the current node is a leaf (it should be, but let's be safe) add it back to the candidates as a 
                # valid result, otherwise discard the branch (by not adding the node back)
                if current_node.is_leaf:
                    # add the remaining size of the current word to the cost
                    # (number of deletions to go from the current word to a true word)
                    candidate_cost_node_word_set.add(
                        (current_cost + len(current_word_left), current_node, "", current_correct_word))

            else:
                # --- previously reccursive cases ---
                # edit is removing the first character
                for child in current_node.children.values():
                    candidate_cost_node_word_set.add(
                        (current_cost + 1, child, current_word_left, current_correct_word + child.value))
                
                # edit is inserting the first character
                candidate_cost_node_word_set.add(
                    (current_cost + 1, current_node, current_word_left[1:], current_correct_word))
                
                # edit is replacing the first character
                # no edit needed on the last character
                for child in current_node.children.values():
                    candidate_cost_node_word_set.add(
                        # +1 to the cost if a replace is needed, +0 if no replace is needed
                        (current_cost + (1 if child.value != current_word_left[0] else 0),
                        child, current_word_left[1:], current_correct_word + child.value))

        if verbose:
            print("Stopped because " + 
                ("we find enough valid final nodes" if len(optimal_cost_node_word_list) >= number_of_results else  
                ("we have no more candidates" if len(candidate_cost_node_word_set) <= 0 else  
                "we have no more candidates under the cost threshold")))

        return list(optimal_cost_node_word_list)


if __name__ == "__main__":
    import datetime
    trie = Trie()
    text = ("A very large number of published documents contain text only. They" +
    "often look boring, and they are often written in obscure language, using " +
    "mile-long sentences and cryptic technical terms, using one font only, " +
    "perhaps even without headings. Such style, or lack of style, might be the" +
    " one you are strongly expected to follow when writing scientific or " +
    "technical reports, legal documents, or administrative papers. It is " +
    "natural to think that such documents would benefit from a few " +
    "illustrative images. (However, just adding illustration might be rather" +
    " useless, if the text remains obscure and unstructured.)")

    trie.train_on_text(text)

    spellchecker = TrieSpellchecker(trie)
    print("'wr'")
    start = datetime.datetime.now()
    print(spellchecker.check_correct("wr"), datetime.datetime.now() - start)
    start = datetime.datetime.now()
    print(spellchecker.trie_levenshtein_spellcorrect("wr", 10), datetime.datetime.now() - start)
    print("'w'")
    start = datetime.datetime.now()
    print(spellchecker.check_correct("w"), datetime.datetime.now() - start)
    start = datetime.datetime.now()
    print(spellchecker.trie_levenshtein_spellcorrect("w", 10), datetime.datetime.now() - start)
    print("''")
    start = datetime.datetime.now()
    print(spellchecker.check_correct(""), datetime.datetime.now() - start)
    start = datetime.datetime.now()
    print(spellchecker.trie_levenshtein_spellcorrect("", 10), datetime.datetime.now() - start)
    print("'Kebap'")
    start = datetime.datetime.now()
    print(spellchecker.check_correct("Kebap"), datetime.datetime.now() - start)
    start = datetime.datetime.now()
    print(spellchecker.trie_levenshtein_spellcorrect("Kebap", 10), datetime.datetime.now() - start)
    print("'unstructured'")
    start = datetime.datetime.now()
    print(spellchecker.check_correct("unstructured"), datetime.datetime.now() - start)
    start = datetime.datetime.now()
    print(spellchecker.trie_levenshtein_spellcorrect("unstructured", 10), datetime.datetime.now() - start)
    start = datetime.datetime.now()
    print(spellchecker.trie_levenshtein_spellcorrect("unstructured", 10, 3), datetime.datetime.now() - start)
    print("'unstruqdsctured'")
    start = datetime.datetime.now()
    print(spellchecker.check_correct("unstruqdsctured"), datetime.datetime.now() - start)
    start = datetime.datetime.now()
    print(spellchecker.trie_levenshtein_spellcorrect("unstruqdsctured", 5), datetime.datetime.now() - start)
    print("'admenestrateve'")
    start = datetime.datetime.now()
    print(spellchecker.check_correct("admenestrateve"), datetime.datetime.now() - start)
    start = datetime.datetime.now()
    print(spellchecker.trie_levenshtein_spellcorrect("admenestrateve", 5), datetime.datetime.now() - start)



    print("\n--- Lipsum corpus ---")
    with open("lipsum.txt", "r", encoding="utf8") as f:
        text_lipsum = f.read().replace("\n", " ")
    trie_lipsum = Trie()
    trie_lipsum.train_on_text(text_lipsum)
    spellchecker_lipsum = TrieSpellchecker(trie_lipsum)

    print("'li'")
    start = datetime.datetime.now()
    print(spellchecker_lipsum.check_correct("li"), datetime.datetime.now() - start)
    start = datetime.datetime.now()
    print(spellchecker_lipsum.trie_levenshtein_spellcorrect("li", 10), datetime.datetime.now() - start)
    print("'l'")
    start = datetime.datetime.now()
    print(spellchecker_lipsum.check_correct("l"), datetime.datetime.now() - start)
    start = datetime.datetime.now()
    print(spellchecker_lipsum.trie_levenshtein_spellcorrect("l", 10), datetime.datetime.now() - start)
    print("''")
    start = datetime.datetime.now()
    print(spellchecker_lipsum.check_correct(""), datetime.datetime.now() - start)
    start = datetime.datetime.now()
    print(spellchecker_lipsum.trie_levenshtein_spellcorrect("", 10), datetime.datetime.now() - start)
    print("'Kebap'")
    start = datetime.datetime.now()
    print(spellchecker_lipsum.check_correct("Kebap"), datetime.datetime.now() - start)
    start = datetime.datetime.now()
    print(spellchecker_lipsum.trie_levenshtein_spellcorrect("Kebap", 10), datetime.datetime.now() - start)
