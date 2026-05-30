"""Capsule-Convex Polygon Overlap task."""

TASK = {
    "title": "Capsule-Convex Polygon Overlap",
    "difficulty": "Hard",
    "function_name": "capsule_poly_overlap",
    "hint": "Decompose into three cases: (1) Is either spine endpoint inside the polygon? For a convex polygon in CCW order, a point P is inside iff cross(edge_i, P-v_i) >= 0 for all edges — check all V edges in one vectorised step. (2) Does the spine segment cross any polygon edge? Use the cross-product orientation test (same primitive as problem #49). (3) Is the minimum distance from the spine to any polygon edge less than r? Reuse the parametric segment-to-segment distance formula from problem #56, vectorised over V edges.",
    "tests": [
        {
            "name": "Capsule spine entirely inside polygon",
            "code": """
import numpy as np
square = np.array([[0.0,0.0],[4.0,0.0],[4.0,4.0],[0.0,4.0]])
capsule = np.array([1.0, 1.0, 3.0, 3.0, 0.2])  # spine from (1,1)→(3,3), r=0.2
assert {fn}(capsule, square) == True, "Spine inside polygon → overlap"
""",
        },
        {
            "name": "Capsule spine crosses polygon boundary",
            "code": """
import numpy as np
square = np.array([[0.0,0.0],[2.0,0.0],[2.0,2.0],[0.0,2.0]])
# Spine from (-1,1)→(3,1) — crosses left and right edges
capsule = np.array([-1.0, 1.0, 3.0, 1.0, 0.1])
assert {fn}(capsule, square) == True, "Spine crossing boundary → overlap"
""",
        },
        {
            "name": "Capsule fully outside, too far from polygon",
            "code": """
import numpy as np
square = np.array([[0.0,0.0],[2.0,0.0],[2.0,2.0],[0.0,2.0]])
# Spine far to the right — closest point on spine to polygon is at x=4
capsule = np.array([4.0, 0.5, 6.0, 0.5, 0.5])  # dist to edge x=2 is 2, r=0.5
assert {fn}(capsule, square) == False, "Separated capsule → no overlap"
""",
        },
        {
            "name": "Capsule radius just reaches a polygon vertex",
            "code": """
import numpy as np
# Triangle with vertex at (0, 0)
tri = np.array([[0.0,0.0],[2.0,0.0],[1.0,2.0]])
# Horizontal spine at y=-1, segment x in [0.5, 1.5]; closest spine point to (0,0) is (0.5,-1), dist=sqrt(0.25+1)≈1.118
# Use r=1.2 to cover it
capsule_reach = np.array([0.5, -1.0, 1.5, -1.0, 1.2])
assert {fn}(capsule_reach, tri) == True, "Radius reaches vertex → overlap"
# Same spine with r=0.9 — does not reach
capsule_miss  = np.array([0.5, -1.0, 1.5, -1.0, 0.9])
assert {fn}(capsule_miss, tri) == False, "Radius too small → no overlap"
""",
        },
        {
            "name": "50-pair mixed correctness and timing test",
            "code": """
import numpy as np, time

def _inside_convex(pt, poly):
    edges = np.roll(poly, -1, axis=0) - poly
    vecs  = pt - poly
    cross = edges[:,0]*vecs[:,1] - edges[:,1]*vecs[:,0]
    return bool((cross >= 0).all())

def _seg_dist(p1, p2, p3, p4):
    EPS = 1e-12
    d1 = p2 - p1; d2 = p4 - p3; rv = p1 - p3
    a = d1@d1; e = d2@d2; b = d1@d2; c = d1@rv; f = d2@rv
    denom = a*e - b*b
    s = float(np.clip((b*f - c*e)/denom, 0, 1)) if denom > EPS else 0.0
    t = float(np.clip((b*s + f)/e, 0, 1)) if e > EPS else 0.0
    s = float(np.clip((b*t - c)/a, 0, 1)) if a > EPS else 0.0
    return float(np.linalg.norm(p1 + s*d1 - p3 - t*d2))

def reference(capsule, poly):
    p1 = capsule[:2]; p2 = capsule[2:4]; r = capsule[4]
    if _inside_convex(p1, poly) or _inside_convex(p2, poly):
        return True
    V = len(poly)
    for i in range(V):
        if _seg_dist(p1, p2, poly[i], poly[(i+1)%V]) <= r:
            return True
    return False

rng = np.random.default_rng(7)
def rand_square(rng):
    x, y = rng.uniform(0, 8), rng.uniform(0, 8)
    s = rng.uniform(1, 3)
    return np.array([[x,y],[x+s,y],[x+s,y+s],[x,y+s]])

t0 = time.time()
for _ in range(50):
    poly = rand_square(rng)
    pts = rng.uniform(0, 10, (2, 2))
    r = float(rng.uniform(0.2, 2.0))
    cap = np.array([pts[0,0], pts[0,1], pts[1,0], pts[1,1], r])
    expected = reference(cap, poly)
    got = {fn}(cap, poly)
    assert got == expected, f"Mismatch: expected {expected}, got {got}"
elapsed = time.time() - t0
assert elapsed < 3.0, f"Too slow: {elapsed:.2f}s"
""",
        },
    ],
}
