"""Line-Polygon Intersections task."""

TASK = {
    "title": "Line-Polygon Intersections",
    "difficulty": "Medium",
    "function_name": "line_polygon_intersections",
    "hint": "For each polygon edge (v1→v2), solve the 2x2 system: origin + t*dir = v1 + s*(v2-v1). Cramer's rule: det = cross(dir, edge); t = cross(v1-origin, edge)/det; s = cross(v1-origin, dir)/det. Include the intersection iff 0<=s<=1 (s is the edge parameter) and det != 0. Vectorise over all V edges at once using numpy, then sort results by t.",
    "tests": [
        {
            "name": "Horizontal line through center of unit square → 2 points",
            "code": """
import numpy as np
square = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
origin = np.array([0.5, 0.5])
direction = np.array([1.0, 0.0])
result = {fn}(origin, direction, square)
result = np.array(result, dtype=float)
assert result.shape[0] == 2, f"Should have 2 intersections, got {result.shape[0]}"
# One at x<0.5 (the left wall) and one at x>0.5 (the right wall), y=0.5
assert all(abs(result[:, 1] - 0.5) < 1e-9), "All intersections at y=0.5"
xs = sorted(result[:, 0])
assert abs(xs[0] - 0.0) < 1e-9 and abs(xs[1] - 1.0) < 1e-9, f"x coords: {xs}"
""",
        },
        {
            "name": "Line parallel to a side — no intersection",
            "code": """
import numpy as np
square = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
origin    = np.array([0.0, 2.0])   # above the square
direction = np.array([1.0, 0.0])   # horizontal — parallel to top/bottom sides
result = {fn}(origin, direction, square)
result = np.array(result)
assert result.shape[0] == 0, f"Parallel line above square should have 0 intersections, got {result.shape[0]}"
""",
        },
        {
            "name": "Diagonal line through square → 2 points, sorted by t",
            "code": """
import numpy as np
square = np.array([[0.0,0.0],[2.0,0.0],[2.0,2.0],[0.0,2.0]])
origin    = np.array([1.0, 1.0])  # center
direction = np.array([1.0, 1.0])  # diagonal
result = {fn}(origin, direction, square)
result = np.array(result, dtype=float)
assert result.shape[0] == 2, f"Diagonal should hit 2 edges, got {result.shape[0]}"
# First intersection has smaller t → smaller x and y
assert result[0, 0] < result[1, 0], "Results should be sorted by t (increasing x)"
""",
        },
        {
            "name": "Line entirely outside polygon → 0 intersections",
            "code": """
import numpy as np
square = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
origin    = np.array([5.0, 0.5])   # far to the right
direction = np.array([0.0, 1.0])   # vertical, does not reach the square
result = {fn}(origin, direction, square)
result = np.array(result)
assert result.shape[0] == 0, f"Line outside should have 0 intersections, got {result.shape[0]}"
""",
        },
        {
            "name": "Large V=5000 polygon — vectorised over all edges",
            "code": """
import numpy as np, time
n = 5000
angles = np.linspace(0, 2*np.pi, n, endpoint=False)
poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
origin    = np.array([0.0, 0.0])
direction = np.array([1.0, 0.0])
t0 = time.time()
result = {fn}(origin, direction, poly)
elapsed = time.time() - t0
result = np.array(result, dtype=float)
assert result.shape[0] == 2, f"Should hit circle at x=-1 and x=1, got {result.shape[0]} pts"
assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s (should vectorise over edges)"
""",
        },
    ],
}
