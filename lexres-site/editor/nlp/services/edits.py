from ...utils.nlp import SpanEdit
from ...utils.models import get_spellchecker
from ...utils.tokenizer import tokenize
import typing as t
"""
    This script should contain only functions for edition proposals.
    Functions should all share the same signature.
"""

VERBOSE = True

def edit_bar_function(text="", verbose=VERBOSE):
    """
        Example function. To be removed
    """

    spellchecker = get_spellchecker("moliere_corneille")

    # tokenize the text
    tokens_and_id = tokenize(text, output_start_and_end_id=True)

    # spellcheck all the tokens
    edits = []
    for token, (start_id, end_id) in tokens_and_id:
        suggestion = spellchecker.check_correct(token).strip('_')
        if suggestion != token:
            edits.append(SpanEdit(beg_idx=start_id, end_idx=end_id, edit=suggestion))

    if verbose: print(f"{len(edits)} fautes d'ortographe détectées")
    return edits