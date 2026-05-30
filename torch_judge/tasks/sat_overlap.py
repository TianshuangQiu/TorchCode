"""SAT Convex Polygon Overlap task."""

TASK = {
    "title": "SAT Convex Polygon Overlap",
    "difficulty": "Medium",
    "function_name": "sat_overlap",
    "hint": "Use the Separating Axis Theorem: two convex polygons do NOT overlap iff there exists an axis (edge normal of either polygon) such that their projections onto that axis are disjoint. Collect all edge normals with np.roll (cyclic vertex shift) + perp rotation, then project both vertex sets with a single matrix multiply (vertices @ axes.T), take min/max per axis, and check for separation.",
    "tests": [
        {
            "name": "Identical unit squares — full overlap",
            "code": """
import numpy as np
square = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
assert {fn}(square, square.copy()) == True, "Identical squares must overlap"
""",
        },
        {
            "name": "Non-overlapping squares — clear separation",
            "code": """
import numpy as np
a = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
b = np.array([[5.0,0.0],[6.0,0.0],[6.0,1.0],[5.0,1.0]])
assert {fn}(a, b) == False, "Squares separated on X should not overlap"
b2 = np.array([[0.0,5.0],[1.0,5.0],[1.0,6.0],[0.0,6.0]])
assert {fn}(a, b2) == False, "Squares separated on Y should not overlap"
""",
        },
        {
            "name": "Half-overlapping squares",
            "code": """
import numpy as np
# a: [0,2]x[0,2], b: [1,3]x[0,2] — intersect in [1,2]x[0,2]
a = np.array([[0.0,0.0],[2.0,0.0],[2.0,2.0],[0.0,2.0]])
b = np.array([[1.0,0.0],[3.0,0.0],[3.0,2.0],[1.0,2.0]])
assert {fn}(a, b) == True, "Half-overlapping squares must overlap"
""",
        },
        {
            "name": "Triangle vertex inside square; square and triangle just separated",
            "code": """
import numpy as np
square = np.array([[0.0,0.0],[4.0,0.0],[4.0,4.0],[0.0,4.0]])
# Triangle with apex well inside the square
tri_in = np.array([[1.0,1.0],[3.0,1.0],[2.0,3.0]])
assert {fn}(square, tri_in) == True, "Triangle inside square → overlap"
# Triangle entirely to the right with 0.01 gap
tri_out = np.array([[4.01,0.0],[6.0,0.0],[5.0,2.0]])
assert {fn}(square, tri_out) == False, "Triangle just outside square → no overlap"
""",
        },
        {
            "name": "100 random convex-polygon pairs — symmetry and timing",
            "code": """
import numpy as np, time

def rand_convex(rng, n_verts=5, scale=3.0):
    angles = np.sort(rng.uniform(0, 2*np.pi, n_verts))
    r = rng.uniform(0.5, scale, n_verts)
    cx, cy = rng.uniform(0, 10), rng.uniform(0, 10)
    return np.stack([cx + r*np.cos(angles), cy + r*np.sin(angles)], axis=1)

rng = np.random.default_rng(99)
t0 = time.time()
for _ in range(100):
    pa = rand_convex(rng)
    pb = rand_convex(rng)
    r1 = {fn}(pa, pb)
    r2 = {fn}(pb, pa)
    assert r1 == r2, f"sat_overlap must be symmetric: {r1} vs {r2}"
elapsed = time.time() - t0
assert elapsed < 3.0, f"Too slow: {elapsed:.2f}s"
""",
        },
    ],
}
