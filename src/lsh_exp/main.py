import logging
import argparse
import numpy as np
from sklearn import datasets, model_selection
from .lsh import LSH


logger = logging.getLogger(__name__)


def calc_ave_precision(ys_pred_label, target):
    ys_pred_label = ys_pred_label.reshape(-1)
    is_pred_correct = ys_pred_label == target

    if np.sum(is_pred_correct) == 0:
        return 0

    precisions = np.cumsum(is_pred_correct) / np.arange(1, len(is_pred_correct) + 1)
    recalls = np.cumsum(is_pred_correct) / np.sum(is_pred_correct)

    diff_recalls = np.diff(recalls, prepend=0)
    avg_precision = np.sum(precisions * diff_recalls)
    return avg_precision


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("hash_length", type=int)
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--num_nearest_vectors", type=int, default=200)
    parser.add_argument("--max_radius", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger.info("start fetch mnist_784")
    digits = datasets.fetch_openml("mnist_784")
    logger.info("done fetch mnist_784")

    n_samples = len(digits.data)
    X = digits.data.reshape((n_samples, -1))
    ys = digits.target
    X_train, X_test, ys_train, ys_test = model_selection.train_test_split(X, ys, train_size=args.train_size, test_size=args.test_size)

    lsh = LSH(args.hash_length)
    lsh.fit(X_train, ys_train)

    avg_precision_lst = []
    for i, (xs_test, y_test) in enumerate(zip(X_test, ys_test), 1):
        if i % 100 == 0:
            logger.info(f"search {i} / {args.test_size}")

        y_pred_cand = lsh.predict(xs_test, num_nearest_vectors=args.num_nearest_vectors)
        avg_precision = calc_ave_precision(y_pred_cand, y_test)
        avg_precision_lst.append(avg_precision)
    mean_average_precision = np.mean(avg_precision_lst)
    print(f"mAP: {mean_average_precision}")
