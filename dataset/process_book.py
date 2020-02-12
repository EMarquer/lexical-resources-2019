
from nltk.tokenize import sent_tokenize
from urllib.request import urlopen, HTTPError
import spacy
import re
import os

GUTENBERG_BOOK_URL = 'https://www.gutenberg.org/files/{}/{}.txt'
BOOK_FOLDER = 'books'
NLP = spacy.load('fr_core_news_sm')
TOKENIZE = False

def spacy_tokenize(book_sentences):
    tokenized_sentences=[]
    for sentence in book_sentences:
        doc = NLP(sentence)
        tokenized_sentences.append([token.text for token in doc])
    return tokenized_sentences

def download_book(book_id, urlpath = GUTENBERG_BOOK_URL):
    """Télécharge un livre du Projet Gutenberg en gérant les possibles problèmes d'encodage"""
    book_content = []

    for extension, encoding in [('-0', "utf-8"), ('', 'ascii'), ('-8', 'ISO 8859-1')]: 
        # iterate through likely filename endings and associated encodings on PG
        try: 
            target_url = urlpath.format(book_id, book_id + extension)
            with urlopen(target_url) as response: 
                for line in response: 
                    # urlopen reads as bytes, to ease processing, we decode to string.
                    # most PG .txt files are encoded in latin-1/ascii format. 
                    try:
                        book_content.append(line.decode(encoding))
                    except: # revert to latin-1 in the event of unexpected PG encoding behaviour 
                        book_content.append(line.decode("latin-1"))
                response.close()
                del response
        except HTTPError: 
            continue
    
    return book_content

def clean_book(book_content, precise_clean=True):
    """
    Cette fonctionalité a pour but de faire du nettoyage.
    Les livres sont tous susceptibles de contenir des informations relatives au Projet Gutenberg.
    En effet, nous avons considéré cela comme étant un bruit.   
    """
    # remove PG metadata precisely, but slower to execute
    if precise_clean == True: 
        start_index = 0  # index for the start of the text
        stop_index = -1  # index for the end of the text  

        two_third_marker = round(len(book_content)*0.67)

        #1. search for *END tags from the back of the file, for two-thirds of the file
        for index_num in range(two_third_marker):
            if re.match(r'\*+\s*END ', book_content[-index_num]):
                stop_index = -index_num

        #2. search for anomalous *START tags in the last two-thirds of the file, 
        for index_num in range(-stop_index, two_third_marker):
            # searching for the last * END from the back, in the last half of the file 
            if re.match(r'\*+\s*START ', book_content[-index_num]):
                stop_index = -index_num

        #3. finally, search for the last START tag from the front, within the first two-thirds
        for index_num in range(two_third_marker):
            # searching for the last * START in the first half of the file 
            if re.match(r'\*+\s*START ', book_content[index_num]):
                start_index = index_num 

        book_content = book_content[start_index:stop_index]

    clean_book_content = " ".join(
        [striped_line for striped_line in (line.strip() for line in book_content) if striped_line])

    # use nltk's sent_tokenise
    all_sentences_in_book = sent_tokenize(clean_book_content)

    # remove 10% of the sentences from the beginning of the book and 10 others from the end, just to be sure
    ten_percent_length = round(len(all_sentences_in_book)*0.10)
    all_sentences_in_book = all_sentences_in_book[ten_percent_length:-ten_percent_length]

    return all_sentences_in_book

if __name__ == "__main__":
    import json
    from pprint import pprint

    # load a list of all the books of the authors
    from select_movements_and_authors import MOVEMENT_AUTHOR_FILE
    with open(MOVEMENT_AUTHOR_FILE, 'r', encoding='utf8') as f:
        movement_author_book_dict = json.load(f)
    pprint(movement_author_book_dict)

    for mvt, authors in movement_author_book_dict.items():
        for author, books in authors.items():
            for book in books:
                print(f"processing book {book} form {author} of the movement {mvt}")
                book_data = download_book(book)
                sentences = clean_book(book_data)

                if TOKENIZE:
                    sentences_tokenized = spacy_tokenize(sentences)
                    
                    path = os.path.join(BOOK_FOLDER, book + ".txt")
                    print(f"savinging to {path}")
                    with open(path, 'w', encoding='utf8') as f:
                        f.write('\n'.join([' '.join(sentence) for sentence in sentences_tokenized]))
                else:
                    path = os.path.join(BOOK_FOLDER, book + ".txt")
                    print(f"savinging to {path}")
                    with open(path, 'w', encoding='utf8') as f:
                        f.write('\n'.join(sentences))
