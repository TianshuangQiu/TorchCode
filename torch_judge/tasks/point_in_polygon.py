"""Point in Polygon task."""

TASK = {
    "title": "Point in Polygon",
    "difficulty": "Medium",
    "function_name": "point_in_polygon",
    "hint": "Ray casting: for each point, count how many polygon edges a rightward horizontal ray crosses. Odd = inside. Vectorise over all P points and V edges simultaneously using broadcasting: points shape (P,1) vs edge values shape (1,V). Key condition: does the edge straddle the ray's y? If yes, check if the crossing x > point's x.",
    "tests": [
        {
            "name": "Center of unit square → True, outside → False",
            "code": """
import numpy as np
square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
points = np.array([[0.5, 0.5], [2.0, 0.5], [-0.5, 0.5]])
result = {fn}(points, square)
assert result.shape == (3,), f"Shape: {result.shape}"
assert result[0] == True,  "Center of square should be inside"
assert result[1] == False, "Right of square should be outside"
assert result[2] == False, "Left of square should be outside"
""",
        },
        {
            "name": "Concave polygon (L-shape) — notch is outside",
            "code": """
import numpy as np
# L-shape: a 3x3 square with the top-right 1x1 corner removed
L = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0],
              [2.0, 2.0], [2.0, 3.0], [0.0, 3.0]])
inside_pt  = np.array([[1.0, 1.0]])   # in the bottom portion
notch_pt   = np.array([[2.5, 2.5]])   # in the removed corner
outside_pt = np.array([[4.0, 1.0]])   # right of the shape
result = {fn}(np.vstack([inside_pt, notch_pt, outside_pt]), L)
assert result[0] == True,  f"Point inside L: {result[0]}"
assert result[1] == False, f"Point in notch of L: {result[1]}"
assert result[2] == False, f"Point outside L: {result[2]}"
""",
        },
        {
            "name": "Equilateral triangle — interior and exterior points",
            "code": """
import numpy as np
tri = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.464]])
pts = np.array([[2.0, 1.0], [2.0, -1.0], [5.0, 1.0]])
result = {fn}(pts, tri)
assert result[0] == True,  "Centroid-ish inside triangle"
assert result[1] == False, "Below triangle"
assert result[2] == False, "Right of triangle"
""",
        },
        {
            "name": "All points outside a tiny polygon",
            "code": """
import numpy as np
tiny = np.array([[10.0, 10.0], [10.1, 10.0], [10.1, 10.1], [10.0, 10.1]])
pts = np.zeros((5, 2))  # all at origin
result = {fn}(pts, tiny)
assert result.shape == (5,)
assert not result.any(), f"All points outside: {result}"
""",
        },
        {
            "name": "Large P=5000 points, V=60-sided polygon — must be fast",
            "code": """
import numpy as np, time
rng = np.random.default_rng(42)
n = 60
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
pts = rng.uniform(-1.5, 1.5, (5000, 2))
t0 = time.time()
result = {fn}(pts, poly)
elapsed = time.time() - t0
assert result.shape == (5000,), f"Shape: {result.shape}"
dists = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
assert (result[dists < 0.95]).all(),   "Points clearly inside circle should be inside polygon"
assert not (result[dists > 1.05]).any(), "Points clearly outside circle should be outside polygon"
assert elapsed < 3.0, f"Too slow: {elapsed:.2f}s (expected <3s)"
""",
        },
    ],
}
