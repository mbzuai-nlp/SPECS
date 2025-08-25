import numpy as np
import torch


def positive_increments(similarity, samples_margin=5):
    if len(similarity.shape) == 3:
        similarity = similarity[:, :, 0]
    n_samples = sum(similarity > 0)

    increments = similarity[:, 1:] - similarity[:, :-1]
    increments = sum(increments > 0)

    results = increments / n_samples[1:]

    drop_filter = n_samples > samples_margin
    return results[drop_filter[1:]]


def negative_increments(similarity):
    sim = similarity[:, :, 0]
    neg_sim = similarity[:, :, 1]
    n_samples = sum(sim > 0)

    increments = neg_sim - sim
    increments = sum(increments > 0)

    results = increments / n_samples

    drop_filter = n_samples > 5
    results = results[drop_filter]
    return results


def increment_rate(similarity):
    sim = similarity[:, :, 0]
    neg_sim = similarity[:, :, 1]
    n_samples = sum(sim > 0)
    n_samples = np.array([value if value > 0 else 1 for value in n_samples])

    neg_increments = neg_sim - sim
    neg_increments = sum(neg_increments > 0)

    increments = sim[:, 1:] - sim[:, :-1]
    increments = sum(increments > 0)

    pos = increments / n_samples[1:]
    neg = neg_increments / n_samples

    drop_filter = n_samples > 5

    pos = pos[drop_filter[1:]]
    neg = neg[drop_filter]

    rate = (pos + 1 - neg[:-1]) / 2

    return pos, neg, rate


def eval_metric(pos_similarities, neg_similarities):
    pos_differences = pos_similarities[1:] - pos_similarities[:1]
    neg_differences = neg_similarities[1:] - neg_similarities[:1]
    increments = pos_differences > 0
    reductions = neg_differences < 0
    det_increments = torch.logical_and(increments, reductions)
    return (increments, reductions, det_increments)


def compute_increments(similarity):
    pos_similarities = similarity[:, :, 0]
    neg_similarities = similarity[:, :, 1]
    mask = pos_similarities > 0
    positives = []
    negatives = []
    for sim, neg, m in zip(pos_similarities, neg_similarities, mask):
        n = sum(m) - 1
        if n > 1:
            pos_diff = sim[1:] - sim[:-1]
            pos_diff = pos_diff > 0
            positives.append(sum(pos_diff[:n]) / n)

            neg_diff = sim - neg
            neg_diff = neg_diff > 0
            negatives.append(sum(neg_diff[:n]) / n)
        else:
            continue
    return positives, negatives
