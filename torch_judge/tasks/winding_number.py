"""Winding Number task."""

TASK = {
    "title": "Winding Number",
    "difficulty": "Medium",
    "function_name": "winding_number",
    "hint": "For each directed edge (v1→v2) and each query point p: upward crossings (v1.y<=p.y<v2.y) add +1 if p is to the LEFT of the edge; downward crossings (v2.y<=p.y<v1.y) add -1 if p is to the RIGHT. 'Left of edge' = cross(v2-v1, p-v1) > 0. Vectorise: broadcast points (P,1,2) vs edge endpoints (1,V,2).",
    "tests": [
        {
            "name": "CCW square: inside → +1, outside → 0",
            "code": """
import numpy as np
# CCW square
ccw_square = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
pts = np.array([[0.5, 0.5],   # inside
                [2.0, 0.5],   # outside right
                [-0.5, 0.5]]) # outside left
result = {fn}(pts, ccw_square)
assert result.shape == (3,), f"Shape: {result.shape}"
assert result[0] ==  1, f"Inside CCW: winding should be +1, got {result[0]}"
assert result[1] ==  0, f"Outside right: winding should be 0, got {result[1]}"
assert result[2] ==  0, f"Outside left: winding should be 0, got {result[2]}"
""",
        },
        {
            "name": "CW square: inside → -1",
            "code": """
import numpy as np
# CW (clockwise) — reverse of CCW
cw_square = np.array([[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]])
pts = np.array([[0.5, 0.5]])
result = {fn}(pts, cw_square)
assert result[0] == -1, f"Inside CW: winding should be -1, got {result[0]}"
""",
        },
        {
            "name": "Non-zero winding means inside regardless of sign",
            "code": """
import numpy as np
ccw = np.array([[0.0,0.0],[2.0,0.0],[2.0,2.0],[0.0,2.0]])
cw  = ccw[::-1].copy()
inside_pt = np.array([[1.0, 1.0]])
r_ccw = {fn}(inside_pt, ccw)
r_cw  = {fn}(inside_pt, cw)
assert r_ccw[0] != 0, "Inside CCW should have non-zero winding"
assert r_cw[0]  != 0, "Inside CW should have non-zero winding"
assert r_ccw[0] == -r_cw[0], "CCW and CW should have opposite winding numbers"
""",
        },
        {
            "name": "Batch of 20 points — both inside and outside",
            "code": """
import numpy as np
n = 12
angles = np.linspace(0, 2*np.pi, n, endpoint=False)
poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # unit circle CCW
rng = np.random.default_rng(0)
pts = rng.uniform(-1.5, 1.5, (20, 2))
result = {fn}(pts, poly)
assert result.shape == (20,), f"Shape: {result.shape}"
dists = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
assert (result[dists < 0.85] != 0).all(), "Clearly inside points should have non-zero winding"
assert (result[dists > 1.15] == 0).all(), "Clearly outside points should have zero winding"
""",
        },
        {
            "name": "Large P=5000, V=50-gon — must be fast",
            "code": """
import numpy as np, time
rng = np.random.default_rng(7)
n = 50
angles = np.linspace(0, 2*np.pi, n, endpoint=False)
poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
pts  = rng.uniform(-1.5, 1.5, (5000, 2))
t0 = time.time()
result = {fn}(pts, poly)
elapsed = time.time() - t0
assert result.shape == (5000,)
assert elapsed < 3.0, f"Too slow: {elapsed:.2f}s (expected <3s)"
""",
        },
    ],
}
