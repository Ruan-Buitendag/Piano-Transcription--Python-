import numpy as np


def stats(ref_intervals, ref_pitches, est_pitches, est_intervals):

    matched = matching(ref_intervals, ref_pitches, est_pitches, est_intervals)

    precision = float(len(matched)) / len(est_pitches)
    recall = float(len(matched)) / len(ref_pitches)

    if precision + recall == 0:
        f_measure = 0
    else:
        f_measure = 2 * precision * recall / (precision + recall)


    return precision, recall, f_measure, len(matched)


def matching(ref_intervals, ref_pitches, est_pitches, est_intervals, pitch_tolerance=50, onset_tolerance=0.05, ):
    onset_distances = np.abs(np.subtract.outer(ref_intervals[:, 0],
                                               est_intervals[:, 0]))
    # Round distances to a target precision to avoid the situation where
    # if the distance is exactly 50ms (and strict=False) it erroneously
    # doesn't match the notes because of precision issues.
    onset_distances = np.around(onset_distances, decimals=4)
    onset_hit_matrix = np.less_equal(onset_distances, onset_tolerance)

    # check for pitch matches
    pitch_distances = np.abs(1200 * np.subtract.outer(np.log2(ref_pitches),
                                                      np.log2(est_pitches)))
    pitch_hit_matrix = np.less_equal(pitch_distances, pitch_tolerance)

    # check for overall matches
    note_hit_matrix = onset_hit_matrix * pitch_hit_matrix
    hits = np.where(note_hit_matrix)

    # Construct the graph input
    # Flip graph so that 'matching' is a list of tuples where the first item
    # in each tuple is the reference note index, and the second item is the
    # estimated note index.
    G = {}
    for ref_i, est_i in zip(*hits):
        if est_i not in G:
            G[est_i] = []
        G[est_i].append(ref_i)

    # bg = BipartiteGraph(len(hits[0]), len(hits[1]))

    # bg = BipartiteGraph(1000, 1000)
    #
    # for ref_i, est_i in zip(*hits):
    #     bg.add_edge(ref_i, est_i)

    # Compute the maximum matching
    matched = sorted(bipartite_match(G).items())

    # aaaa = len(matched)
    #
    # bbbb = bg.maximum_matching()

    return matched


def bipartite_match(graph):
    """Find maximum cardinality matching of a bipartite graph (U,V,E).
    The input format is a dictionary mapping members of U to a list
    of their neighbors in V.

    The output is a dict M mapping members of V to their matches in U.

    Parameters
    ----------
    graph : dictionary : left-vertex -> list of right vertices
        The input bipartite graph.  Each edge need only be specified once.

    Returns
    -------
    matching : dictionary : right-vertex -> left vertex
        A maximal bipartite matching.

    """
    # Adapted from:
    #
    # Hopcroft-Karp bipartite max-cardinality matching and max independent set
    # David Eppstein, UC Irvine, 27 Apr 2002

    # initialize greedy matching (redundant, but faster than full search)
    matching = {}
    for u in graph:
        for v in graph[u]:
            if v not in matching:
                matching[v] = u
                break

    while True:
        # structure residual graph into layers
        # pred[u] gives the neighbor in the previous layer for u in U
        # preds[v] gives a list of neighbors in the previous layer for v in V
        # unmatched gives a list of unmatched vertices in final layer of V,
        # and is also used as a flag value for pred[u] when u is in the first
        # layer
        preds = {}
        unmatched = []
        pred = dict([(u, unmatched) for u in graph])
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)

        # repeatedly extend layering structure by another pair of layers
        while layer and not unmatched:
            new_layer = {}
            for u in layer:
                for v in graph[u]:
                    if v not in preds:
                        new_layer.setdefault(v, []).append(u)
            layer = []
            for v in new_layer:
                preds[v] = new_layer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)

        # did we finish layering without finding any alternating paths?
        if not unmatched:
            unlayered = {}
            for u in graph:
                for v in graph[u]:
                    if v not in preds:
                        unlayered[v] = None
            return matching

        def recurse(v):
            """Recursively search backward through layers to find alternating
            paths.  recursion returns true if found path, false otherwise
            """
            if v in preds:
                L = preds[v]
                del preds[v]
                for u in L:
                    if u in pred:
                        pu = pred[u]
                        del pred[u]
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return True
            return False

        for v in unmatched:
            recurse(v)

class BipartiteGraph:
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.graph = [[] for _ in range(left)]

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def maximum_matching(self):
        def dfs(u):
            for v in self.graph[u]:
                if not visited[v]:
                    visited[v] = True
                    if right_match[v] == -1 or dfs(right_match[v]):
                        left_match[u] = v
                        right_match[v] = u
                        return True
            return False

        left_match = [-1] * self.left
        right_match = [-1] * self.right
        match_count = 0

        for u in range(self.left):
            visited = [False] * self.right
            if dfs(u):
                match_count += 1

        return match_count

# # Example usage:
# if __name__ == "__main__":
#     left = 4
#     right = 4
#     bg = BipartiteGraph(left, right)
#     bg.add_edge(0, 1)
#     bg.add_edge(0, 2)
#     bg.add_edge(1, 3)
#     bg.add_edge(2, 3)
#
#     max_matching = bg.maximum_matching()
    # print("Maximum cardinality bipartite matching:", max_matching)
