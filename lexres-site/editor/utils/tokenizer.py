import re

TOKEN_RE = r"([\w-]+)"
SENTENCE_SPLIT_RE = r"[\.;!\?~\*]"

def tokenize(text: str, output_start_and_end_id: bool=False):
    if output_start_and_end_id:
        return [(match.group(), match.span()) for match in re.finditer(TOKEN_RE, text)]
    else:
        tokens = re.findall(TOKEN_RE, text)

def sent_tokenize(text: str):
    return re.split(SENTENCE_SPLIT_RE, text)