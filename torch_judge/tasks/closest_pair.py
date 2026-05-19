"""Closest Pair of Points task."""

TASK = {
    "title": "Closest Pair of Points",
    "difficulty": "Hard",
    "function_name": "closest_pair",
    "hint": "Divide and conquer — O(N log N). (1) Sort by x. (2) Recurse on left and right halves to get delta = min(d_left, d_right). (3) Collect points within delta of the midline into a strip; sort strip by y. (4) For each strip point, check only the next 7 strip points — classic proof shows at most 8 per delta x delta rectangle. Return (distance, i, j) with i < j.",
    "tests": [
        {
            "name": "Two points — trivial case",
            "code": """
import numpy as np
pts = np.array([[0.0, 0.0], [3.0, 4.0]])
dist, i, j = {fn}(pts)
assert abs(dist - 5.0) < 1e-9, f"Distance: {dist}"
assert i == 0 and j == 1, f"Indices: ({i}, {j})"
""",
        },
        {
            "name": "Three points — correct pair identified",
            "code": """
import numpy as np
pts = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
dist, i, j = {fn}(pts)
assert abs(dist - 1.0) < 1e-9, f"Closest distance should be 1.0, got {dist}"
assert set([i, j]) == {0, 1}, f"Indices should be (0,1), got ({i},{j})"
""",
        },
        {
            "name": "Grid of points — matches brute-force",
            "code": """
import numpy as np
rng = np.random.default_rng(42)
pts = rng.uniform(-10, 10, (60, 2))
# Brute force reference
best = float('inf'); bi = bj = -1
for ii in range(len(pts)):
    for jj in range(ii+1, len(pts)):
        d = np.linalg.norm(pts[ii] - pts[jj])
        if d < best:
            best = d; bi, bj = ii, jj
dist, i, j = {fn}(pts)
assert abs(dist - best) < 1e-9, f"Distance mismatch: {dist} vs {best}"
assert set([i,j]) == set([bi,bj]), f"Index mismatch: ({i},{j}) vs ({bi},{bj})"
""",
        },
        {
            "name": "Duplicate points → distance = 0.0",
            "code": """
import numpy as np
pts = np.array([[1.0, 2.0], [5.0, 3.0], [1.0, 2.0], [9.0, 8.0]])
dist, i, j = {fn}(pts)
assert abs(dist - 0.0) < 1e-12, f"Duplicate points: distance should be 0, got {dist}"
assert i < j, f"Index constraint i < j violated: i={i}, j={j}"
""",
        },
        {
            "name": "Large N=10000 — must beat O(N^2)",
            "code": """
import numpy as np, time
rng = np.random.default_rng(99)
pts = rng.uniform(-1000, 1000, (10_000, 2))
t0 = time.time()
dist, i, j = {fn}(pts)
elapsed = time.time() - t0
assert dist > 0, "Distance should be positive for random points"
assert 0 <= i < j < 10_000, f"Invalid indices: ({i},{j})"
assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s — O(N^2) brute force would take ~10s"
""",
        },
    ],
}
