from datasets import load_from_disk
import random
from nltk.tokenize import word_tokenize
from nltk import pos_tag


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


def compare_text(sentence, detail):
    tokens = word_tokenize(detail)
    pos_tags = pos_tag(tokens)
    nouns = [word for word, pos in pos_tags if pos in ("NN", "NNS")]
    return any(noun in sentence for noun in nouns)


def main():
    dataset = load_from_disk(
        "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/ln_nouns_v2.hf"
    )
    dataset = dataset.remove_columns(["neg_details"])

    # def generate_splits(examples):
    #     captions = split_by(examples["caption"], ["NN"])
    #     return {"captions": captions}

    # dataset = dataset.map(generate_splits)

    # # save here in only interested in positive details.
    # dataset.save_to_disk(
    #     "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/ln_nouns.hf"
    # )

    def create_neg_splits(example):
        neg_details = []
        for i in range(len(example["captions"]) - 1):
            while True:
                sample = random.choice(dataset)
                if sample["image"] != example["image"]:
                    if i >= len(sample["captions"]):
                        i = len(sample["captions"]) - 1
                    if i > 0:
                        neg_det = (
                            sample["captions"][i]
                            .replace(sample["captions"][i - 1], "")
                            .strip()
                        )
                    else:
                        neg_det = sample["captions"][i].strip()
                    if not compare_text(example["caption"], neg_det):
                        break

            neg_details.append(neg_det)
        return {"neg_details": neg_details}

    dataset = dataset.map(create_neg_splits)

    def merge_neg(example):
        neg_captions = [
            example["captions"][i] + " " + example["neg_details"][i]
            for i in range(len(example["neg_details"]))
        ]
        return {"neg_captions": neg_captions}

    dataset = dataset.map(merge_neg).remove_columns(["neg_details"])

    dataset.save_to_disk(
        "/leonardo_scratch/fast/EUHPC_D12_071/clipfinecap/data/ln_nouns.hf"
    )


if __name__ == "__main__":
    main()
