from collections import Counter

AUTHOR_MOVEMENT_FILE = "author_movements.json" # file with for each author, the movements
MOVEMENT_AUTHOR_FILE = "movements_author_books.json" # file with the selected movements and the selected authors
UNKNOWN_MOVEMENT = "mouvement inconnu"
N_MOVEMENTS = 4
N_AUTHORS = 3

def movement_authors(author_mvt_dict):
    mvt_authors=dict()
    for author, movements in author_mvt_dict.items():
        for movement in movements:
            if movement not in mvt_authors.keys(): mvt_authors[movement]=[]
            mvt_authors[movement].append(author)
    return mvt_authors

def filter_movements_and_authors(author_mvt_dict, author_book_dict, n_movements = 3, n_authors = 3):
    mvt_authors_dict = movement_authors(author_mvt_dict)

    # get the number of authors per movement
    mvt_author_counter = Counter()
    for mvt, authors in mvt_authors_dict.items():
        if mvt != UNKNOWN_MOVEMENT: # ignore the unknown movements
            mvt_author_counter[mvt] = len(authors)

    # get the correct number of movements
    movements = [movement for movement, count in mvt_author_counter.most_common(n_movements)]

    # get the top most furnished authors of each movement
    final_movements_authors_dict = dict()
    for movement in movements:
        authors = mvt_authors_dict[movement]
        
        # get the number of book per author
        author_book_counter = Counter()
        for author in authors: author_book_counter[author] = len([book for book in author_book_dict[author] if book])

        # get the authors with the most books
        authors = [author for author, count in author_book_counter.most_common(n_authors)]

        # assemble the whole data
        final_movements_authors_dict[movement] = {
            # author_book_dict[author] contains a dict of the shape {book title: book id}, but we only need 'book id'
            author: list(author_book_dict[author].values())
            for author in authors
        }

    return final_movements_authors_dict






if __name__ == "__main__":
    import json
    from pprint import pprint
    with open(AUTHOR_MOVEMENT_FILE, 'r', encoding='utf8') as f:
        author_mvt_dict = json.load(f)

    from get_author_book_list import AUTHOR_BOOK_FILE
    with open(AUTHOR_BOOK_FILE, 'r', encoding='utf8') as f:
        author_book_dict = json.load(f)

    mvt_authors = filter_movements_and_authors(author_mvt_dict, author_book_dict, N_MOVEMENTS, N_AUTHORS)
    pprint(mvt_authors)

    with open(MOVEMENT_AUTHOR_FILE, 'w', encoding='utf8') as f:
        json.dump(mvt_authors, f, indent = 4)