import logging
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations


logger = logging.getLogger(__name__)


class LSH:
    def __init__(self, hash_length):
        self.hash_length = hash_length

    def fit(self, X_train, ys_train):
        """
        Args:
            X_train:
                (n, dim)
        """

        self.X_train = X_train
        n, dim = X_train.shape

        self.X_train = np.copy(X_train)
        self.W = np.random.randn(dim, self.hash_length)  # (d, m)
        hashes = self.__convert_vecs_to_hashes(X_train)
        buckets = self.__convert_hashes_to_buckets(hashes)

        # bucket_idx -> (label, count)
        bucket_idx_to_counter = defaultdict(Counter)

        for train_idx, bucket_idx in zip(np.arange(n), buckets):
            bucket_idx_to_counter[bucket_idx][ys_train[train_idx]] += 1

        self.bucket_idx_to_counter = bucket_idx_to_counter

    def predict(self, vec, num_nearest_vectors=200):
        """
        Args:
            vec:
                1-d array
            num_nearest_vectors:
                int
        Return:
            ys_pred_cand: [class_1, class_2, ..., class_n] ( in a certain order )
        """

        _vec = vec.reshape(1, -1)  # (1, dim)
        pred_label_counter = Counter()

        hash_val = self.__convert_vecs_to_hashes(_vec)  # (1, hash_length)
        bucket = self.__convert_hashes_to_buckets(hash_val)[0]
        pred_label_counter.update(self.bucket_idx_to_counter[bucket])

        # search until candidate size > num_nearest_vectors
        for radius in range(1, self.hash_length + 1):
            if sum(pred_label_counter.values()) > num_nearest_vectors:
                break

            inversion_indices_combs = combinations(range(self.hash_length), radius)
            for inversion_indices in inversion_indices_combs:
                search_hash_val = np.copy(hash_val)
                search_hash_val[:, inversion_indices] = abs(search_hash_val[:, inversion_indices] - 1)  # 1 -> |1 - 1| = 0, 0 -> |0 - 1| = 1

                bucket = self.__convert_hashes_to_buckets(search_hash_val)[0]
                pred_label_counter.update(self.bucket_idx_to_counter[bucket])

        ys_pred_cand = [label for label, _ in pred_label_counter.most_common()]
        return np.array(ys_pred_cand)

    def __convert_vecs_to_hashes(self, vecs):
        hashes = (vecs.dot(self.W) > 0).astype(int)  # (n, hash_length)
        return hashes

    def __convert_hashes_to_buckets(self, hashes):
        buckets = hashes.dot(1 << np.arange(self.hash_length))  # (n, )
        return buckets
