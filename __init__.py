from typing import Iterable

from pygraphblas import Matrix, types


def chosen_vertices(indices: Iterable[int], weights: Iterable[int] =None):
    weights = [1 for _ in indices] if weights is None else tuple(weights)
    indices = tuple(indices)  # copy and make a tuple just in case
    return Matrix.from_lists(indices, indices, weights)


def check_matrix_correctness(A: Matrix):
    n, m = A.nrows, A.ncols
    rows, cols, weights = A.to_lists()
    if n != m:
        raise ValueError('Adjacency matrix must have equal number of rows and columns')
    if not (all([0 <= r < n and 0 <= c < n for r, c in zip(rows, cols)])):
        raise ValueError(f'Matrix must operate on vertices 0..{n-1}')
    if not all([w in (0, 1) for w in weights]):  # tuple is faster than set for small collections
        raise ValueError('all weights must be either 0 or 1')


def bfs(A: Matrix, indices: Iterable[int]):
    """
    For simplicity we assume that vertices are numbers 0..n-1 and all weights have value 1.
    Takes adjacency matrix and an iterable of starting vertices.
    Returns tuple (origins, levels).
    origins is a dict: vertex -> starting vertex.
    levels is a dict: vertex -> distance from starting vertices.
    """
    check_matrix_correctness(A)
    A_t = A.transpose()
    Identity = Matrix.identity(types.UINT8, nrows=A_t.nrows)
    indices = tuple(indices)
    O = chosen_vertices(indices)
    origins = {i: i for i in indices}
    levels = {i: 0 for i in indices}
    Cumulative = O
    NotReached = Identity - O
    level = 1
    while Cumulative.max() and NotReached:
        Cumulative = Cumulative @ A_t
        Current = Cumulative @ NotReached
        _, current_cols, current_weights = Current.to_lists()
        NotReached -= chosen_vertices(current_cols, weights=current_weights)
        NotReached = NotReached.nonzero()
        for origin, vertex, weight in zip(*Current.to_lists()):
            if weight:
                origins[vertex] = origin
                levels[vertex] = level
        level += 1
    return origins, levels


if __name__ == '__main__':
    A = Matrix.from_lists([0, 1, 2], [1, 0, 2], [1, 1, 1], ncols=3, nrows=3)
    print(bfs(A, [0, 2]))
