from pygraphblas import Matrix, types

from bfs import bfs


def test_empty():
    A = Matrix.sparse(types.UINT8, nrows=5, ncols=5)
    indices = [0, 2]
    origins, levels = bfs(A, indices)
    assert origins == {0: 0, 2: 2}
    assert levels == {0: 0, 2: 0}


def test_single():
    # 2 -> 1, 1 -> 0, 1 -> 3
    A = Matrix.from_lists([2, 1, 1], [1, 3, 0], [1, 1, 1], nrows=4, ncols=4)
    indices = [2]
    origins, levels = bfs(A, indices)
    assert origins == {i: 2 for i in range(4)}
    assert levels == {2: 0, 1: 1, 0: 2, 3: 2}


def test_cycle():
    A = Matrix.from_lists([0, 1, 2], [1, 2, 0], [1, 1, 1], nrows=3, ncols=3)
    indices = [0]
    origins, levels = bfs(A, indices)
    assert origins == {i: 0 for i in range(3)}
    assert levels == {i: i for i in range(3)}


def test_separate():
    # two separate graphs
    A = Matrix.from_lists([0, 2, 2], [1, 3, 4], [1, 1, 1], nrows=5, ncols=5)
    indices = [0, 2]
    origins, levels = bfs(A, indices)
    assert origins == {0: 0, 1: 0, 2: 2, 3: 2, 4: 2}
    assert levels == {0: 0, 2: 0, 1: 1, 3: 1, 4: 1}


def test_subtree():
    A = Matrix.from_lists([0, 1, 2, 2], [1, 2, 3, 4], [1, 1, 1, 1], nrows=5, ncols=5)
    indices = [0, 2]
    origins, levels = bfs(A, indices)
    assert origins == {0: 0, 1: 0, 2: 2, 3: 2, 4: 2}
    assert levels == {0: 0, 2: 0, 1: 1, 3: 1, 4: 1}


def test_arbitrary():
    A = Matrix.from_lists([0, 1, 2, 2], [2, 2, 3, 4], [1, 1, 1, 1], nrows=5, ncols=5)
    indices = [0, 1]
    origins, levels = bfs(A, indices)
    assert origins == {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
    assert levels == {0: 0, 2: 1, 1: 0, 3: 2, 4: 2}
