from .trie import Trie, TrieSpellchecker, TrieAutocompleter
from .language_model import BackoffMechanism, CharEncoder
import typing as t
import os

THIS_FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

TRIE_FOLDER = "../../../models/trie"
TRIE_SINGLETONS: t.Dict[str, t.Tuple[Trie, TrieSpellchecker, TrieAutocompleter]] = dict()

LM_FOLDER = "../../../models/language_model"
LM_BACKOFF_FILE = "backoff.pkl"
LM_ENCODER_FILE = "encoder.pkl"
LM_SINGLETON = None
LM_ENCODER_SINGLETON = None
MAX_NEXT_WORDS = 1

MODEL_LIST_FILE = "../../../models/list.txt"
MODEL_LIST = None
DEFAULT_MODEL = "théatre français"
CURRENT_MODEL = DEFAULT_MODEL

def get_model_list():
    global MODEL_LIST
    if MODEL_LIST is None:
        with open(os.path.join(THIS_FILE_DIRECTORY, MODEL_LIST_FILE), 'r', encoding="utf8") as f:
            MODEL_LIST = [model.strip() for model in f]
    return MODEL_LIST

def set_model(model):
    global CURRENT_MODEL
    CURRENT_MODEL = model
    print("new model is set to", CURRENT_MODEL)

def get_current_model():
    global CURRENT_MODEL
    return CURRENT_MODEL

def load_trie(model = None):
    if model is None: model = get_current_model()

    global TRIE_SINGLETONS
    if model not in TRIE_SINGLETONS.keys():
        # load a model
        path = os.path.join(THIS_FILE_DIRECTORY, TRIE_FOLDER, model + ".pkl")
        trie = Trie.load(path)
        spellchecker = TrieSpellchecker(trie)
        autocompleter = TrieAutocompleter(trie)
        TRIE_SINGLETONS[model] = trie, spellchecker, autocompleter
        print(f"loaded {model}")

    return TRIE_SINGLETONS[model][0]

def get_spellchecker(model = None):
    if model is None: model = get_current_model()
    
    global TRIE_SINGLETONS
    if model not in TRIE_SINGLETONS.keys():
        # make the trie loader load the correct model
        load_trie(model)
    return TRIE_SINGLETONS[model][1]

def get_autocompleter(model = None):
    if model is None: model = get_current_model()
    
    global TRIE_SINGLETONS
    if model not in TRIE_SINGLETONS.keys():
        # make the trie loader load the correct model
        load_trie(model)
    return TRIE_SINGLETONS[model][2]

def get_language_model(model = None, max_next_words=MAX_NEXT_WORDS):
    if model is None: model = get_current_model()
    
    global LM_SINGLETON, LM_ENCODER_SINGLETON
    # load the backoff model
    if LM_SINGLETON is None:
        LM_SINGLETON = BackoffMechanism.load(
            os.path.join(THIS_FILE_DIRECTORY, LM_FOLDER),
            LM_BACKOFF_FILE)
    # load the character encoder
    if LM_ENCODER_SINGLETON is None:
        LM_ENCODER_SINGLETON = CharEncoder.load(
            os.path.join(THIS_FILE_DIRECTORY, LM_FOLDER, LM_ENCODER_FILE))
    
    return (lambda context: 
        LM_SINGLETON.sample(model, LM_ENCODER_SINGLETON, context, max_next_words))