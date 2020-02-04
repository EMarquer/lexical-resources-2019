# --- imports ---
from pprint import pprint
import re
from SPARQLWrapper import SPARQLWrapper, JSON
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.query()
sparql.addDefaultGraph("http://dbpedia.org")

# --- constants ---
VERBOSE=False

# ontology
SPARQ_AUTHOR_NAME = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX dbp: <http://dbpedia.org/ontology/>
    SELECT ?person
    WHERE {{
        ?person a dbp:Person .
        ?person foaf:name ?name
        FILTER ((LANG(?name)="en" or LANG(?name)="fr") and CONTAINS(?name, "{}")).
    }}
    LIMIT 10
"""
SPARQ_MOVEMENTS = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX dbp: <http://dbpedia.org/ontology/>
    PREFIX dct: <http://purl.org/dc/terms/>
    SELECT ?genre ?genre_name
    WHERE {{
        <{}> dbp:genre ?genre .
        ?genre dct:subject dbc:Literary_movements .
        ?genre rdfs:label ?genre_name .
        FILTER (LANG(?genre_name)="fr")
    }}
    LIMIT 10
"""

# --- code ---
def get_uri_from_name(author_name, verbose=VERBOSE):
    query = SPARQ_AUTHOR_NAME.format(author_name)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    if verbose: print("Query:\n{}".format(query))
    
    results = sparql.query().convert()
        
    if len(results['results']['bindings']) > 0: # if the querry did yield results, return the first one
        author_uri = results['results']['bindings'][0]['person']['value']
        if verbose: print("\nResult: '{}'".format(author_uri))
    
    else: # if the querry didn't yield any result, return None
        author_uri = None
        if verbose: print("\nNo result")
    
    return author_uri

def get_author_uri(author_name, verbose=VERBOSE):
    if verbose: print("Raw name: {}".format(author_name))
    
    # three kind of names (in each case, can be followed by years):
    # - family name, initials (given name)
    # - family name, given name
    # - full name
    re_family_given_parenthesis_date = r'([^,]+), (?:(.+) )?\(([^,]+)\)(, ?[-\d\?]+)?'
    re_family_given_date = r'([^,]+), (?:([^,]+))(, ?[-\d\?]+)?'
    re_date = r'(, ?[-\d\?]+)?'
    
    author_uri = None
    
    # try to match "family name, initials (given name)"
    match = re.match(re_family_given_parenthesis_date, author_name)
    if match:
        family_name, initials, given_name = match.group(1, 2, 3)
        if verbose: print("Given name:  {}\n"
                          "Initials:    {}\n"
                          "Family name: {}".format(given_name, initials, family_name))
            
        # query with given and family names, and with initials and family name if it fails
        author_uri = (get_uri_from_name(given_name + ' ' + family_name, verbose=verbose) or
                      get_uri_from_name(initials + ' ' + family_name, verbose=verbose))
        
    # try to match "family name, given name"
    match = re.match(re_family_given_date, author_name)
    if match and not author_uri:
        family_name, given_name = match.group(1, 2)
        if verbose: print("Given name:  {}\n"
                          "Family name: {}".format(given_name, family_name))
            
        # query with given and family names
        author_uri = get_uri_from_name(given_name + ' ' + family_name, verbose=verbose)
         
    # if nothing else yielded result, we just remove the years at the end of the name
    if not author_uri:
        name = re.sub(re_date, '', author_name)
        if verbose: print("Full name: {}".format(name))
            
        # query with full name
        author_uri = get_uri_from_name(name, verbose=verbose)
        
    return author_uri

def get_movements(author_uri, verbose=VERBOSE):
    query = SPARQ_MOVEMENTS.format(author_uri)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    #if verbose: print("Query:\n{}".format(query))
    
    results = sparql.query().convert()
    
    # create a dictionary of languages and the corresponding abstract
    movements = dict()
    for result in results['results']['bindings']:
        movements[result['genre']['value']] = result['genre_name']['value']
        
    if verbose: print("Movements"); pprint(movements)
    
    return movements

def get_movements_from_name(author_name, verbose=VERBOSE):
    author_uri = get_author_uri(author, verbose=verbose)
    
    if author_uri:
        if verbose: print(author, "found:", author_uri)

        movements = get_movements(author_uri, verbose=verbose)
        
        return author_uri, movements
        
    else:
        if verbose: print(f"'{author}' not found among the authors")
        return None, None

# --- main ---
if __name__ == "__main__":
    test = {
        'Aaron, S. F. (Samuel Francis), 1862-': {
            "Radio Boys Cronies\rOr, Bill Brown's Radio": '/ebooks/11861',
            'Radio Boys Loyalty; Or, Bill Brown Listens In': '/ebooks/25753'},
        'Abbott, Charles C. (Charles Conrad), 1843-1919': {'Outings at Odd Times': '/ebooks/48916',
            'Travels in a Tree-top': '/ebooks/55805'},
        'Abbott, Edwin Abbott, 1838-1926': {'Flatland: A Romance of Many Dimensions': '/ebooks/45506',
            'Flatland: A Romance of Many Dimensions (Illustrated)': '/ebooks/201',
            'How to Write Clearly: Rules and Exercises on English Composition': '/ebooks/22600',
            'Onesimus: Memoirs of a Disciple of St. Paul': '/ebooks/54223',
            'Philochristus: Memoirs of a Disciple of the Lord': '/ebooks/48843',
            'Silanus the Christian': '/ebooks/56843'},
        'Abbott, Eleanor Hallowell, 1872-1958': {'Fairy Prince and Other Stories': '/ebooks/26399',
            'The Indiscreet Letter': '/ebooks/15728',
            'Little Eve Edgarton': '/ebooks/15660',
            'Molly Make-Believe': '/ebooks/18665',
            'Old-Dad': '/ebooks/48990',
            'Peace on Earth, Good-will to Dogs': '/ebooks/20213',
            'Rainy Week': '/ebooks/43025',
            "The Sick-a-Bed LadyAnd Also Hickory Dock, The Very Tired Girl, The Happy-Day, Something That Happened in October, The Amateur Lover, Heart of The City, The Pink Sash, Woman's Only Business": '/ebooks/34829',
            'The Stingy Receiver': '/ebooks/49330',
            'The White Linen Nurse': '/ebooks/14506'},
        'Guy de Maupassant':{}}

    for author in test.keys():
        print(get_movements_from_name(author))