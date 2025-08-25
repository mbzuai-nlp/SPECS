from nltk.tokenize import word_tokenize
from nltk import pos_tag
import random


def split_by(sentence, grammars):
    sentences = []
    current = ""
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    for i, (s, t) in enumerate(pos_tags):
        if current == "":
            current = s
        else:
            current = current + " " + s
        if i == len(tokens) - 1:
            sentences.append(sentence)
        elif any(t.startswith(grammar) for grammar in grammars):
            if not any(pos_tags[i + 1][1].startswith(grammar) for grammar in grammars):
                sentences.append(current)
    return sentences
