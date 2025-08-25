import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag_sents, pos_tag

import torch.nn.functional as F
import torch


def cosine_similarity(embd1, embd2):
    """
    Assuming embd1 is of shape [batch_size1, embedding_dim] and embd2 is of shape [batch_size2, embedding_dim]
    """
    embd1 = F.normalize(embd1, p=2, dim=1)
    embd2 = F.normalize(embd2, p=2, dim=1)
    embd2 = embd2.t()
    similarity = torch.matmul(embd1, embd2)
    return similarity.cpu().numpy()


def generate_sentence_variations(sentence):
    words = sentence.split()
    variations = [words[: i + 1] for i in range(len(words))]
    return [" ".join(variation) for variation in variations]


def make_ungrammatical(sentence, percentage=None):
    if percentage is None:
        n = len(sentence.split())
    else:
        n = int(len(sentence.split()) * percentage)
    tokens = word_tokenize(sentence)
    nwords_swap = np.random.randint(0, len(tokens), n)
    for i in range(len(nwords_swap) - 1):
        tokens[nwords_swap[i]], tokens[nwords_swap[i + 1]] = (
            tokens[nwords_swap[i + 1]],
            tokens[nwords_swap[i]],
        )
    ungrammatical_sentence = " ".join(tokens)
    return ungrammatical_sentence


def eliminate_all_grammars(sentence, grammars):
    sentences = []
    current = ""
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    filtered_words = [s for s, t in pos_tags if t not in grammars]
    current = " ".join(filtered_words)
    return [sentence, current]


def count_grammars(caption, grammar_types):
    tokens = word_tokenize(caption)
    pos_tags = pos_tag(tokens)
    count = 0
    for token, tag in pos_tags:
        if tag in grammar_types:
            count += 1
    return count


def filter_grammars(sentence, grammars):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    filtered_words = [
        s for s, t in pos_tags if any(t.startswith(grammar) for grammar in grammars)
    ]
    return " ".join(filtered_words)


# from utils import positive_increments, negative_increments, compute_increment_rate
# similarity = np.load("results/dci_test_detail_b32.npy")
# print(np.mean(positive_increments(similarity)))
# print(np.mean(negative_increments(similarity)))
# print(np.mean(compute_increment_rate(similarity)))
