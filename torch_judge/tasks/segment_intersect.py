"""Batch Segment Intersection task."""

TASK = {
    "title": "Batch Segment Intersection",
    "difficulty": "Medium",
    "function_name": "segment_intersect",
    "hint": "This is the cross-product orientation test. For segments AB and CD: compute d1=cross(AB,AC), d2=cross(AB,AD), d3=cross(CD,CA), d4=cross(CD,CB). Proper intersection: d1*d2<0 AND d3*d4<0. Handle collinear endpoints separately. Vectorise by broadcasting segs_a[:,None] vs segs_b[None,:] to get shape (N,M,...).",
    "tests": [
        {
            "name": "Perpendicular + segments that should not intersect",
            "code": """
import numpy as np
# Two perpendicular segments forming a + sign
a = np.array([[[0.0, 1.0], [2.0, 1.0]]])  # horizontal
b = np.array([[[1.0, 0.0], [1.0, 2.0]],   # vertical — crosses a
              [[3.0, 0.0], [3.0, 2.0]]])   # to the right — does not cross a
result = {fn}(a, b)
assert result.shape == (1, 2), f"Shape: {result.shape}"
assert result[0, 0] == True,  "Perpendicular segments cross"
assert result[0, 1] == False, "Parallel segments do not cross"
""",
        },
        {
            "name": "Parallel segments — no intersection",
            "code": """
import numpy as np
a = np.array([[[0.0, 0.0], [4.0, 0.0]],
              [[0.0, 1.0], [4.0, 1.0]]])
b = np.array([[[0.0, 2.0], [4.0, 2.0]]])
result = {fn}(a, b)
assert result.shape == (2, 1)
assert result[0, 0] == False, "Parallel — no intersection"
assert result[1, 0] == False, "Parallel — no intersection"
""",
        },
        {
            "name": "T-intersection — endpoint lies on other segment",
            "code": """
import numpy as np
# AB is horizontal; CD's start point C lies at the midpoint of AB
a = np.array([[[0.0, 0.0], [4.0, 0.0]]])
b = np.array([[[2.0, 0.0], [2.0, 3.0]]])  # C is on AB
result = {fn}(a, b)
assert result[0, 0] == True, "T-intersection at endpoint should count"
""",
        },
        {
            "name": "Collinear overlapping segments",
            "code": """
import numpy as np
a = np.array([[[0.0, 0.0], [3.0, 0.0]]])
b = np.array([[[2.0, 0.0], [5.0, 0.0]]])  # collinear, overlapping
result = {fn}(a, b)
assert result[0, 0] == True, "Collinear overlapping segments intersect"
""",
        },
        {
            "name": "Large N=M=300 — loopy solution would be slow",
            "code": """
import numpy as np, time
rng = np.random.default_rng(7)
pts_a = rng.uniform(-10, 10, (300, 2, 2))
pts_b = rng.uniform(-10, 10, (300, 2, 2))
t0 = time.time()
result = {fn}(pts_a, pts_b)
elapsed = time.time() - t0
assert result.shape == (300, 300), f"Shape: {result.shape}"
assert result.dtype == bool or str(result.dtype) == 'bool', f"dtype: {result.dtype}"
assert elapsed < 3.0, f"Too slow: {elapsed:.2f}s (expected <3s — use broadcasting)"
""",
        },
    ],
}
