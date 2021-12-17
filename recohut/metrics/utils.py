# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/metrics/utils.ipynb (unless otherwise specified).

__all__ = ['calculate_precision_recall']

# Cell
import numpy as np
from sklearn.metrics import precision_score, recall_score, ndcg_score

# Cell
def calculate_precision_recall(X, y_true, y_pred, N, threshold):
    """Calculate the precision and recall scores.

    Args:
        X
        y_true
        y_pred
        N
        threshold

    Returns:
        precision_score (float)
        recall_score (float)
    """
    precision = 0
    recall = 0
    count = 0

    rec_true = np.array([1 if rating >= threshold else 0 for rating in y_true])
    rec_pred = np.zeros(y_pred.size)

    for user_id in np.unique(X[:,0]):
        indices = np.where(X[:,0] == user_id)[0]

        rec_true = np.array([1 if y_true[i] >= threshold else 0 for i in indices])

        if (np.count_nonzero(rec_true) > 0): # ignore test users without relevant ratings

            user_pred = np.array([y_pred[i] for i in indices])
            rec_pred = np.zeros(indices.size)

            for pos in np.argsort(user_pred)[-N:]:
                if user_pred[pos] >= threshold:
                    rec_pred[pos] = 1

            precision += precision_score(rec_true, rec_pred, zero_division=0)
            recall += recall_score(rec_true, rec_pred)
            count += 1

    return precision/count, recall/count