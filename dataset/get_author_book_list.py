from urllib.request import urlopen



from urllib.request import urlopen, HTTPError
import urllib, random, spacy, pickle
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import sent_tokenize

DRAMA_URL = 'https://www.gutenberg.org/wiki/FR_Th%C3%A9%C3%A2tre_(Genre)'
AUTHOR_BOOK_FILE = "author_books.json"

def get_url(url):
    """Récupère le contenu de la page récupéré par BeautifulSoup à partir de l'url"""
    html = urlopen(url)
    parsed_page = BeautifulSoup(html, "html.parser")
    return parsed_page

def get_book_id(book_url):
    """Récupère les identifiants des livres à partir de la page correspondante"""
    soup_book = get_url(book_url)
    
    table_extract = soup_book.find("table", {'class': 'files'})
    table_extract_book_link = soup_book.find_all('tr')
    
    book_link_html=table_extract.find('td', {'class':'unpadded icon_save'})

    links = soup_book.find_all("a")
    for a in links:
        if a.get('type') and a.get('type').startswith("text/plain"):
            link = a.get('href')
            if link.startswith('/ebooks/'):
                return link.split("/")[-1].split('.')[0]

    only_book_link = book_link_html.find_all("a")
    for a in only_book_link:   
        link = a.get('href')
        if link.startswith('/ebooks/'):
                return link.split("/")[-1]

def clean_book_url(url):
    if url.startswith('//www.'):
        cleaned_url = 'http://ww' + url[len('www.'):]
        return cleaned_url
    return url

def author_and_book(parsed_page):
    content_div = parsed_page.find('div', class_='mw-parser-output')
    
    if not content_div: # if the main div is not found, return empy list of books
        yield None, []
    
    else:
        authors_and_book_lists = parsed_page.find_all(["ul", "td", "p", "h2"])
        author_name = None
        for elem in authors_and_book_lists:
            if elem.name == 'h2':
                # get the first a tag with name attribute as the author name if there is at least one
                #author_names = [a_tag['id'] for a_tag in elem.find_all('span', {'class': 'mw-headline'}) if a_tag.has_attr('id')]
                author_names = [span.get_text() for span in elem.find_all('span', {'class': 'mw-headline'})]
                author_name = author_names[0] if author_names else None
                #print(name_list)
            elif elem.name in {'ul', 'td', "p"} and author_name:
                book_li_list = elem.find_all('a', class_="extiw")
                books = dict()
                for book_li in book_li_list:
                    if book_li:
                        book_title = book_li.get_text()
                        book_link = clean_book_url(book_li['href'])
                        book_id = get_book_id(book_link)
                            
                        if book_id:
                            # add the book to the list of books
                            books[book_title] = book_id #book_link
                yield author_name, books

                # clean the memory to avoid getting multiple ul per author
                author_name = None    


if __name__ == "__main__":
    # variable to instantiate
    VERBOSE = False
    from pprint import pprint
    
    # extract page
    parsed_page = get_url(DRAMA_URL)
    
    # get all the authors and books
    author_book_dict = {author_name: books for author_name, books in author_and_book(parsed_page)}
    pprint(author_book_dict)

    import json
    with open(AUTHOR_BOOK_FILE, 'w', encoding='utf8') as f:
        json.dump(author_book_dict, f, indent=4)
