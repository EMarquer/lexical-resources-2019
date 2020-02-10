from ...utils.nlp import SpanEdit
from ...utils.models import get_autocompleter
from ...utils.tokenizer import tokenize
import typing as t
"""
    This script should contain only functions for edition proposals.
    Functions should all share the same signature.
"""

MAX_SUGGESTIONS = 5
VERBOSE = True

def complete_bar_function(text="", verbose=VERBOSE, max_suggestions=MAX_SUGGESTIONS):
    """
        Example function. To be removed
    """

    autocompleter = get_autocompleter("moliere_corneille")

    # tokenize the text
    tokens_and_id = tokenize(text, output_start_and_end_id=True)

    # spellcheck all the tokens
    edits = []
    for token, (start_id, end_id) in tokens_and_id:
        # compute the top suggestions
        suggestion = autocompleter.complete(token, max_selected=MAX_SUGGESTIONS)

        # add the suggestion if at least 1 is provided
        if len(suggestion) > 0 and suggestion[0] != token:
            edits.append(SpanEdit(beg_idx=start_id, end_idx=end_id, edit=', '.join(suggestion)))

    if verbose: print(f"{len(edits)} autocompletions suggérées")
    return edits