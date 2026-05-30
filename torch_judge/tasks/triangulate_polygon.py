"""Polygon Triangulation (Ear-Clipping) task."""

TASK = {
    "title": "Polygon Triangulation (Ear-Clipping)",
    "difficulty": "Hard",
    "function_name": "triangulate_polygon",
    "hint": "Use ear-clipping. An ear is a vertex B (with neighbours A, C) such that: (1) the turn A→B→C is convex — cross(B-A, C-A) > 0; and (2) no other active vertex lies inside (or on the boundary of) triangle ABC. Vectorise the inner ear test: reshape active vertices to (K,1,2) and candidate triangles to (1,K,2), compute the three edge cross-products as (K,K) arrays with >= 0, AND them — one broadcast step replaces the O(K) inner loop. Use an exclusion mask to skip the ear's own three vertices (A, B, C), which trivially satisfy >= 0. Clip the first ear found, shrink the active set, repeat.",
    "tests": [
        {
            "name": "Trivial triangle — single output triangle",
            "code": """
import numpy as np
tri = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
result = {fn}(tri)
assert result.shape == (1, 3), f"Shape: {result.shape}"
assert set(result[0]) == {0, 1, 2}, f"Indices: {result[0]}"
""",
        },
        {
            "name": "Convex square — correct count and total area",
            "code": """
import numpy as np
square = np.array([[0.0,0.0],[2.0,0.0],[2.0,2.0],[0.0,2.0]])
result = {fn}(square)
assert result.shape == (2, 3), f"Shape: {result.shape}"
assert result.min() >= 0 and result.max() <= 3, "Indices out of range"
def tri_area(v, t): A,B,C=v[t[0]],v[t[1]],v[t[2]]; return 0.5*abs((B-A)[0]*(C-A)[1]-(B-A)[1]*(C-A)[0])
total = sum(tri_area(square, t) for t in result)
assert abs(total - 4.0) < 1e-9, f"Total area: {total}"
""",
        },
        {
            "name": "Concave L-shaped polygon — correct count and total area",
            "code": """
import numpy as np
# L-shape: (0,0),(4,0),(4,2),(2,2),(2,4),(0,4) CCW — area = 4*4 - 2*2 = 12
L = np.array([[0.0,0.0],[4.0,0.0],[4.0,2.0],[2.0,2.0],[2.0,4.0],[0.0,4.0]])
result = {fn}(L)
assert result.shape == (4, 3), f"Shape: {result.shape} (expected (4,3) for 6-vertex polygon)"
assert result.min() >= 0 and result.max() <= 5, "Indices out of range"
def tri_area(v, t): A,B,C=v[t[0]],v[t[1]],v[t[2]]; return 0.5*abs((B-A)[0]*(C-A)[1]-(B-A)[1]*(C-A)[0])
total = sum(tri_area(L, t) for t in result)
assert abs(total - 12.0) < 1e-9, f"Total area: {total} (expected 12)"
""",
        },
        {
            "name": "Random convex polygon — total area matches shoelace",
            "code": """
import numpy as np
rng = np.random.default_rng(17)
angles = np.sort(rng.uniform(0, 2*np.pi, 12))
verts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
result = {fn}(verts)
assert result.shape == (10, 3), f"Shape: {result.shape}"
assert result.min() >= 0 and result.max() <= 11
def tri_area(v, t): A,B,C=v[t[0]],v[t[1]],v[t[2]]; return 0.5*abs((B-A)[0]*(C-A)[1]-(B-A)[1]*(C-A)[0])
tri_total = sum(tri_area(verts, t) for t in result)
x, y = verts[:,0], verts[:,1]
shoelace = 0.5 * abs(np.sum(x * np.roll(y,-1) - np.roll(x,-1) * y))
assert abs(tri_total - shoelace) < 1e-9, f"Area mismatch: {tri_total} vs {shoelace}"
""",
        },
        {
            "name": "n=200 convex polygon — must complete in <5s",
            "code": """
import numpy as np, time
rng = np.random.default_rng(31)
n = 200
angles = np.sort(rng.uniform(0, 2*np.pi, n))
verts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
t0 = time.time()
result = {fn}(verts)
elapsed = time.time() - t0
assert result.shape == (n-2, 3), f"Shape: {result.shape}"
assert result.min() >= 0 and result.max() <= n-1
def tri_area(v, t): A,B,C=v[t[0]],v[t[1]],v[t[2]]; return 0.5*abs((B-A)[0]*(C-A)[1]-(B-A)[1]*(C-A)[0])
x, y = verts[:,0], verts[:,1]
shoelace = 0.5 * abs(np.sum(x * np.roll(y,-1) - np.roll(x,-1) * y))
tri_total = sum(tri_area(verts, t) for t in result)
assert abs(tri_total - shoelace) < 1e-6, f"Area mismatch: {tri_total} vs {shoelace}"
assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s (expected <5s — vectorise the inner ear test)"
""",
        },
    ],
}
