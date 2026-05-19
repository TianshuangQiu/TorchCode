"""Convex Hull (Graham Scan) task."""

TASK = {
    "title": "Convex Hull (Graham Scan)",
    "difficulty": "Hard",
    "function_name": "convex_hull",
    "hint": "Graham scan: (1) find the bottommost-leftmost pivot, (2) sort remaining points by polar angle using np.arctan2, (3) scan the sorted list maintaining a stack — pop whenever the last 3 points make a right turn (cross product <= 0). The cross product of (B-A) and (C-A) tells you the turn direction.",
    "tests": [
        {
            "name": "Unit square — 4 hull vertices in CCW order",
            "code": """
import numpy as np

def _area(pts):
    x, y = pts[:,0], pts[:,1]
    return 0.5 * abs(np.sum(x * np.roll(y,-1) - np.roll(x,-1) * y))

square = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
result = {fn}(square)
result = np.array(result, dtype=float)
assert len(result) == 4, f"Square hull should have 4 points, got {len(result)}"
assert abs(_area(result) - 1.0) < 1e-9, f"Hull area: {_area(result)}"
""",
        },
        {
            "name": "Triangle — all 3 points are on the hull",
            "code": """
import numpy as np
tri = np.array([[0.0,0.0],[4.0,0.0],[2.0,3.0]])
result = {fn}(tri)
result = np.array(result, dtype=float)
assert len(result) == 3, f"Triangle hull should have 3 points, got {len(result)}"
""",
        },
        {
            "name": "Interior points filtered out — only hull points returned",
            "code": """
import numpy as np
rng = np.random.default_rng(99)
# Place 4 corners of a square plus 20 interior points
corners = np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
interior = rng.uniform(0.5, 9.5, (20, 2))
points = np.vstack([corners, interior])
result = {fn}(points)
result = np.array(result, dtype=float)
assert len(result) == 4, f"Only 4 corners should be on hull, got {len(result)}"

def _area(pts):
    x, y = pts[:,0], pts[:,1]
    return 0.5 * abs(np.sum(x * np.roll(y,-1) - np.roll(x,-1) * y))

assert abs(_area(result) - 100.0) < 1e-6, f"Hull area: {_area(result)}"
""",
        },
        {
            "name": "Regular hexagon — all 6 points on hull",
            "code": """
import numpy as np
n = 6
angles = np.linspace(0, 2*np.pi, n, endpoint=False)
hex_pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
rng = np.random.default_rng(0)
interior = rng.uniform(-0.4, 0.4, (10, 2))
pts = np.vstack([hex_pts, interior])
result = {fn}(pts)
result = np.array(result, dtype=float)
assert len(result) == 6, f"Hexagon hull should have 6 points, got {len(result)}"
""",
        },
        {
            "name": "Large N=50000 — must complete quickly",
            "code": """
import numpy as np, time
rng = np.random.default_rng(42)
pts = rng.standard_normal((50_000, 2))
t0 = time.time()
result = {fn}(pts)
elapsed = time.time() - t0
result = np.array(result, dtype=float)
assert result.ndim == 2 and result.shape[1] == 2, f"Shape: {result.shape}"
assert len(result) >= 3, "Hull needs at least 3 points"
assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s (expected <5s)"
""",
        },
    ],
}
