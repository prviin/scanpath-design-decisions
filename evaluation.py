import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import cdist, euclidean
import re

EPS = np.finfo(np.float32).eps


class EvaluationFunctions:
    @staticmethod
    def dtw(p, q):
        """
        Compute Dynamic Time Warping distance between two Numpy arrays.
        """
        # If dist is an int of value p > 0, then the p-norm will be used.
        # Check https://github.com/slaypni/fastdtw/blob/ea13ae2e6761f056623ff1d3ac6c71fcf4f94497/fastdtw/fastdtw.py#L93
        # for more details
        dist, _ = fastdtw(p, q, dist=2)
        return dist

    @staticmethod
    def tde(p, q, k=2, distance_mode="Mean"):
        """
        Compute Time Delay Embedding distance between two Numpy arrays.
        See https://github.com/dariozanca/FixaTons/
        """
        # k must be shorter than both lists lenghts
        if len(p) < k or len(q) < k:
            raise ValueError(
                "ERROR: Too large value for the time-embedding vector dimension,"
                f"k must be shorter than both lists lenghts but {k=}, {len(p)=}, {len(q)=}"
            )

        # create time-embedding vectors for both scanpaths
        p_vectors = []
        for i in np.arange(0, len(p) - k + 1):
            p_vectors.append(p[i : i + k])
        q_vectors = []
        for i in np.arange(0, len(q) - k + 1):
            q_vectors.append(q[i : i + k])

        distances = []

        # in the following loops, for each k-vector from the simulated scan-path
        # we look for the k-vector from humans, the one of minimum distance,
        # and we save the value of such a distance, divided by k
        for s_k_vec in q_vectors:
            norms = []
            for h_k_vec in p_vectors:
                d = np.linalg.norm(s_k_vec - h_k_vec)
                norms.append(d)
            distances.append(min(norms) / k)

        # At this point, the list "distances" contains the value of
        # minimum distance for each simulated k-vec
        # according to the distance_mode. Here we compute the similarity
        # between the two scan-paths.
        if distance_mode == "Mean":
            return sum(distances) / len(distances)
        elif distance_mode == "Hausdorff":
            return max(distances)

        raise ValueError(f"Unknown distance mode: {distance_mode}.")

    #  def eyenalysis(p, q):
    #    """
    #    Compute the eyenalysis distance between two Numpy arrays of the same length.

    #    Let dist(MxN) be the two-dimensional pair-wise
    #    Euclidean distance of every fixation in p with all other fixations in Q
    #    """
    #    dist = cdist(p, q, 'euclidean') ** 2
    #    return (dist.min(axis=0).sum() + dist.min(axis=1).sum()) / (max(p.shape[0], q.shape[0]

    @staticmethod
    def eyenalysis(p, q):
        """
        Compute the eyenalysis distance between two Numpy arrays of the same length.
        """

        # dist = np.zeros((p.shape[0], q.shape[0]))
        # for idx_1, fix_1 in np.ndenumerate(p):
        #   for idx_2, fix_2 in np.ndenumerate(q):
        #       dist[idx_1, idx_2] = euclidean(fix_1, fix_2)
        dist = cdist(p, q, "euclidean") ** 2

        return (1 / (p.shape[0] + q.shape[0])) * (
            dist.min(axis=0).sum() + dist.min(axis=1).sum()
        )

    @staticmethod
    def __coincidence_matrix(p, q, threshold=0.05):
        """
        Compute alignment matrix between two Numpy sequences of the same length.
        """
        min_len = p.shape[0] if (p.shape[0] < q.shape[0]) else q.shape[0]
        p = p[:min_len, :2]
        q = q[:min_len, :2]

        assert p.shape == q.shape
        s = p.shape[0]
        c = np.zeros((s, s))

        for i in range(s):
            for j in range(s):
                if euclidean(p[i], q[j]) < threshold:
                    c[i, j] = 1
        return c, min_len

    @staticmethod
    def recurrence(p, q, threshold=0.05):
        """
        Compute Recurrence distance between two Numpy arrays.
        See https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
        """
        c, min_len = EvaluationFunctions.__coincidence_matrix(p, q, threshold)
        r = np.triu(c, 1).sum()

        return 100 * (2 * r) / (min_len * (min_len - 1))

    @staticmethod
    def corm(p, q, threshold=0.05):
        """
        Compute Center of recurrence mass (CORM) between two Numpy arrays.
        See https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
        """
        c, min_len = EvaluationFunctions.__coincidence_matrix(p, q, threshold)
        r = np.triu(c, 1).sum() + EPS

        counter = 0

        for i in range(0, min_len - 1):
            for j in range(i + 1, min_len):
                counter += (j - i) * c[i, j]

        return 100 * (counter / ((min_len - 1) * r))

    @staticmethod
    def determinism(p, q, threshold=0.05):
        """
        Compute Determinism distance between two Numpy arrays.
        See https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
        """
        c, min_len = EvaluationFunctions.__coincidence_matrix(p, q, threshold)
        r = np.triu(c, 1).sum() + EPS

        counter = 0
        for i in range(1, min_len):
            data = c.diagonal(i)
            data = "".join([str(int(item)) for item in data])
            similar_subsequences = re.findall("1{2,}", data)
            for seq in similar_subsequences:
                counter += len(seq)

        return 100 * (counter / r)

    @staticmethod
    def laminarity(p, q, threshold=0.05):
        """
        Compute Laminarity distance between two Numpy arrays.
        See https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
        """
        min_len = p.shape[0] if (p.shape[0] < q.shape[0]) else q.shape[0]
        p = p[:min_len, :2]
        q = q[:min_len, :2]

        c, _ = EvaluationFunctions.__coincidence_matrix(p, q, threshold)
        R = np.triu(c, 1).sum() + EPS
        # print(c, np.triu(c, 1), sep="\n")
        C = np.sum(c)
        # print(C)

        c = c.astype(int)

        # print(R)

        HL = 0
        HV = 0

        for i in range(min_len):
            data = c[i, :]
            data = "".join([str(item) for item in data])
            similar_subsequences = re.findall("1{2,}", data)
            for seq in similar_subsequences:
                HL += len(seq)

        for j in range(min_len):
            data = c[:, j]
            data = "".join([str(item) for item in data])
            similar_subsequences = re.findall("1{2,}", data)
            for seq in similar_subsequences:
                HV += len(seq)

        return 100 * ((HL + HV) / (2 * C))


if __name__ == "__main__":
    p = np.array([[1, 2], [3, 4], [1, 2], [1, 2], [7, 8], [9, 10]]) / 10
    q = np.array([[1, 2], [1, 2], [1, 3], [6, 7], [8, 9]]) / 10
    # print("dtw", EvaluationFunctions.dtw(p, q))
    # print("tde", EvaluationFunctions.tde(p, q))
    # print("eyenalysis", EvaluationFunctions.eyenalysis(p, q))
    # print("recurrence", EvaluationFunctions.recurrence(p, q))
    # print("corm", EvaluationFunctions.corm(p, q))
    # print("determinism", EvaluationFunctions.determinism(p, q))
    # print("laminarity", EvaluationFunctions.laminarity(p, q))
    # print(EvaluationFunctions.laminarity(q, q))
    # print(EvaluationFunctions.laminarity(p, p))
    for func_name, func in EvaluationFunctions.__dict__.items():
        if not func_name.startswith("_") and callable(func.__func__):
            print(func_name, func(p, q))
