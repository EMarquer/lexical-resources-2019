from gensim.utils import tokenize
from gensim.models.fasttext import FastText
import re

if __name__ == "__main__":
    
    moliere_files = [f"molière_{i}.txt" for i in range(1,5)]
    corneille_files = [f"corneille_{i}.txt" for i in range(1,8)]
    moliere_files += corneille_files

    def read_files(file_pathes, splitter = r"[\.;!\?~\*]"):
        for path in file_pathes:
            with open(path, 'r', encoding="utf8") as f:
                sentences = [sentence.strip() for sentence in re.split(splitter, f.read()) if sentence]
                for sentence in sentences:
                    yield list(tokenize(sentence))

    model = FastText(size=128)
    print("building vocab")
    model.build_vocab(sentences=read_files(moliere_files))
    print("training")
    model.train(sentences=read_files(moliere_files), epochs=model.epochs,
        total_examples=model.corpus_count, total_words=model.corpus_total_words)

    print("similarity")
    print(model.wv.most_similar("nuit"))
    print(model.wv.most_similar("nuot"))
    print(model.wv.most_similar("bonjour"))
    print(model.wv.most_similar(("bomjour","bomjour","bomjour")))