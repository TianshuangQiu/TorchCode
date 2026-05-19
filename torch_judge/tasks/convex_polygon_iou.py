"""Convex Polygon IoU task."""

TASK = {
    "title": "Convex Polygon IoU",
    "difficulty": "Medium",
    "function_name": "convex_polygon_iou",
    "hint": "IoU = intersection_area / union_area = inter / (area_a + area_b - inter). To find the intersection polygon of two convex polygons, use Sutherland-Hodgman clipping (clip poly_a against each half-plane of poly_b). Then compute the shoelace area of the result.",
    "tests": [
        {
            "name": "Identical polygons → IoU = 1.0",
            "code": """
import numpy as np
square = np.array([[0.0,0.0],[2.0,0.0],[2.0,2.0],[0.0,2.0]])
result = {fn}(square, square.copy())
assert abs(result - 1.0) < 1e-6, f"Identical polygons: IoU should be 1.0, got {result}"
""",
        },
        {
            "name": "Non-overlapping polygons → IoU = 0.0",
            "code": """
import numpy as np
a = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
b = np.array([[5.0,5.0],[6.0,5.0],[6.0,6.0],[5.0,6.0]])
result = {fn}(a, b)
assert abs(result - 0.0) < 1e-9, f"Non-overlapping: IoU should be 0.0, got {result}"
""",
        },
        {
            "name": "Half-overlapping squares → IoU = 1/3",
            "code": """
import numpy as np
# Square A [0,2] x [0,2], area=4
# Square B [1,3] x [0,2], area=4
# Intersection [1,2] x [0,2], area=2 → IoU = 2/(4+4-2) = 2/6 = 1/3
a = np.array([[0.0,0.0],[2.0,0.0],[2.0,2.0],[0.0,2.0]])
b = np.array([[1.0,0.0],[3.0,0.0],[3.0,2.0],[1.0,2.0]])
result = {fn}(a, b)
assert abs(result - 1/3) < 1e-6, f"Half-overlap: IoU should be 1/3, got {result:.6f}"
""",
        },
        {
            "name": "IoU is symmetric: iou(a,b) == iou(b,a)",
            "code": """
import numpy as np
a = np.array([[0.0,0.0],[3.0,0.0],[3.0,1.0],[0.0,1.0]])
b = np.array([[1.0,-1.0],[4.0,-1.0],[4.0,2.0],[1.0,2.0]])
r1 = {fn}(a, b)
r2 = {fn}(b, a)
assert abs(r1 - r2) < 1e-9, f"IoU not symmetric: {r1} vs {r2}"
assert 0 <= r1 <= 1, f"IoU out of [0,1]: {r1}"
""",
        },
        {
            "name": "100 random polygon pairs — all results in [0,1]",
            "code": """
import numpy as np, time
rng = np.random.default_rng(13)
def rand_square(rng):
    x, y = rng.uniform(0, 5), rng.uniform(0, 5)
    s = rng.uniform(0.5, 2)
    return np.array([[x,y],[x+s,y],[x+s,y+s],[x,y+s]])

t0 = time.time()
for _ in range(100):
    a = rand_square(rng)
    b = rand_square(rng)
    r = {fn}(a, b)
    assert 0 <= r <= 1 + 1e-6, f"IoU out of range: {r}"
elapsed = time.time() - t0
assert elapsed < 3.0, f"Too slow: {elapsed:.2f}s for 100 pairs"
""",
        },
    ],
}
