"""Batch AABB Overlap task."""

TASK = {
    "title": "Batch AABB Overlap",
    "difficulty": "Easy",
    "function_name": "rect_overlap",
    "hint": "This is a broadcasting comparison. Reshape rects_a to (N,1,4) and rects_b to (1,M,4), then check the Separating Axis Theorem: two AABBs overlap iff NEITHER axis separates them — i.e., a.x2 > b.x1 AND a.x1 < b.x2 AND a.y2 > b.y1 AND a.y1 < b.y2.",
    "tests": [
        {
            "name": "Basic: overlapping and non-overlapping pairs",
            "code": """
import numpy as np
rects_a = np.array([[0.0, 0.0, 2.0, 2.0],
                    [5.0, 5.0, 7.0, 7.0]])
rects_b = np.array([[1.0, 1.0, 3.0, 3.0],
                    [6.0, 0.0, 8.0, 2.0]])
result = {fn}(rects_a, rects_b)
assert result.shape == (2, 2), f"Shape: {result.shape}"
assert result[0, 0] == True,  "a[0] and b[0] overlap"
assert result[0, 1] == False, "a[0] and b[1] do not overlap (separated on Y)"
assert result[1, 0] == False, "a[1] and b[0] do not overlap (separated on X)"
assert result[1, 1] == False, "a[1] and b[1] do not overlap (separated on Y)"
""",
        },
        {
            "name": "Separation on X axis",
            "code": """
import numpy as np
a = np.array([[0.0, 0.0, 1.0, 1.0]])
b = np.array([[2.0, 0.0, 3.0, 1.0]])  # to the right, gap between 1 and 2
result = {fn}(a, b)
assert result.shape == (1, 1)
assert result[0, 0] == False, "Separated on X — no overlap"
""",
        },
        {
            "name": "Separation on Y axis",
            "code": """
import numpy as np
a = np.array([[0.0, 0.0, 1.0, 1.0]])
b = np.array([[0.0, 2.0, 1.0, 3.0]])  # above, gap between 1 and 2
result = {fn}(a, b)
assert result[0, 0] == False, "Separated on Y — no overlap"
""",
        },
        {
            "name": "Touching at edge — strict > convention means False",
            "code": """
import numpy as np
a = np.array([[0.0, 0.0, 1.0, 1.0]])
b = np.array([[1.0, 0.0, 2.0, 1.0]])  # share the edge x=1 only
result = {fn}(a, b)
assert result[0, 0] == False, "Touching at edge only — strict > means no interior overlap"
""",
        },
        {
            "name": "Large N=M=3000 — loopy solution would be slow",
            "code": """
import numpy as np, time
rng = np.random.default_rng(0)
xy1 = rng.uniform(0, 100, (3000, 2))
wh  = rng.uniform(1, 10, (3000, 2))
rects_a = np.concatenate([xy1, xy1 + wh], axis=1)
xy1 = rng.uniform(0, 100, (3000, 2))
wh  = rng.uniform(1, 10, (3000, 2))
rects_b = np.concatenate([xy1, xy1 + wh], axis=1)
t0 = time.time()
result = {fn}(rects_a, rects_b)
elapsed = time.time() - t0
assert result.shape == (3000, 3000), f"Shape: {result.shape}"
assert result.dtype == bool or result.dtype == np.bool_, f"dtype: {result.dtype}"
assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s (expected <5s — use broadcasting)"
""",
        },
    ],
}
