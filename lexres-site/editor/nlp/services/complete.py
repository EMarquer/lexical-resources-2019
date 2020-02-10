from ...utils.nlp import SpanEdit
from ...utils.models import get_autocompleter
from ...utils.tokenizer import tokenize
import typing as t
"""
    This script should contain only functions for edition proposals.
    Functions should all share the same signature.
"""

VERBOSE = True

def complete_bar_function(text="", verbose=VERBOSE):
    """
        Example function. To be removed
    """

    autocompleter = get_autocompleter("moliere_corneille")

    # tokenize the text
    tokens_and_id = tokenize(text, output_start_and_end_id=True)

    # spellcheck all the tokens
    edits = []
    for token, (start_id, end_id) in tokens_and_id:
        suggestion = autocompleter.complete(token)
        if len(suggestion) > 0 and suggestion[0] != token:
            edits.append(SpanEdit(beg_idx=start_id, end_idx=end_id, edit=suggestion))

    if verbose: print(f"{len(edits)} autocompletions suggérées")
    return edits