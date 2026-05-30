"""Batch Capsule-Capsule Overlap task."""

TASK = {
    "title": "Batch Capsule-Capsule Overlap",
    "difficulty": "Medium",
    "function_name": "capsule_overlap",
    "hint": "Reduce to segment-to-segment minimum distance. Parametrise both spines: P(s)=P1+s*d1, Q(t)=P3+t*d2 with s,t in [0,1]. Minimise ||P(s)-Q(t)||^2: solve the 2x2 linear system for unconstrained (s,t), clamp s to [0,1], recompute t, clamp t, recompute s. Vectorise by broadcasting caps_a[:,None,:] vs caps_b[None,:,:] — all dot-products become (N,M) arrays.",
    "tests": [
        {
            "name": "Touching capsules — spines parallel, radii sum equals gap",
            "code": """
import numpy as np
# Two horizontal capsules, spines at y=0 and y=3, each with radius 1.5
# distance between spines = 3.0 = r1+r2 → just touching (True with <=)
a = np.array([[0.0, 0.0, 2.0, 0.0, 1.5]])   # spine y=0, r=1.5
b = np.array([[0.0, 3.0, 2.0, 3.0, 1.5]])   # spine y=3, r=1.5
result = {fn}(a, b)
assert result.shape == (1, 1), f"Shape: {result.shape}"
assert result[0, 0] == True, "Touching capsules (dist==r_sum) should overlap"
""",
        },
        {
            "name": "Separated capsules — spines parallel, gap exceeds radii sum",
            "code": """
import numpy as np
a = np.array([[0.0, 0.0, 2.0, 0.0, 0.5]])   # spine y=0, r=0.5
b = np.array([[0.0, 5.0, 2.0, 5.0, 0.5]])   # spine y=5, r=0.5 → gap=5 > 1
result = {fn}(a, b)
assert result.shape == (1, 1)
assert result[0, 0] == False, "Separated capsules should not overlap"
""",
        },
        {
            "name": "Perpendicular crossing capsules",
            "code": """
import numpy as np
# Horizontal and vertical capsules whose spines cross at (1,1) — overlap
a = np.array([[0.0, 1.0, 3.0, 1.0, 0.1]])   # horizontal, r=0.1
b = np.array([[1.0, 0.0, 1.0, 3.0, 0.1]])   # vertical,   r=0.1
result = {fn}(a, b)
assert result[0, 0] == True, "Crossing spines → overlap"
# Non-crossing perpendicular pair
a2 = np.array([[0.0, 0.0, 2.0, 0.0, 0.3]])   # horizontal y=0
b2 = np.array([[5.0, -1.0, 5.0, 1.0, 0.3]])  # vertical x=5 — far away
result2 = {fn}(a2, b2)
assert result2[0, 0] == False, "Non-crossing perpendicular pair should not overlap"
""",
        },
        {
            "name": "Degenerate capsule (zero-length spine = circle)",
            "code": """
import numpy as np
# Point capsule at origin with r=2; circle at (1,0) with r=2 → overlap (dist=1 < 4)
a = np.array([[0.0, 0.0, 0.0, 0.0, 2.0]])
b = np.array([[1.0, 0.0, 1.0, 0.0, 2.0]])
r = {fn}(a, b)
assert r[0, 0] == True, "Two overlapping circles (degenerate capsules)"
# Far apart: dist=10 > 2+2
c = np.array([[10.0, 0.0, 10.0, 0.0, 1.0]])
r2 = {fn}(a, c)
assert r2[0, 0] == False, "Separated circles"
""",
        },
        {
            "name": "Large N=M=500 — loop solution would be slow",
            "code": """
import numpy as np, time
rng = np.random.default_rng(42)
pts_a = rng.uniform(-50, 50, (500, 4))
pts_b = rng.uniform(-50, 50, (500, 4))
r_a = rng.uniform(0.5, 3.0, (500, 1))
r_b = rng.uniform(0.5, 3.0, (500, 1))
caps_a = np.hstack([pts_a, r_a])
caps_b = np.hstack([pts_b, r_b])
t0 = time.time()
result = {fn}(caps_a, caps_b)
elapsed = time.time() - t0
assert result.shape == (500, 500), f"Shape: {result.shape}"
assert result.dtype == bool or str(result.dtype) == 'bool', f"dtype: {result.dtype}"
assert elapsed < 3.0, f"Too slow: {elapsed:.2f}s (expected <3s — use broadcasting)"
""",
        },
    ],
}
