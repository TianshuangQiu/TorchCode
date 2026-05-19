"""Polygon Clip (Sutherland-Hodgman) task."""

TASK = {
    "title": "Polygon Clip (Sutherland-Hodgman)",
    "difficulty": "Hard",
    "function_name": "polygon_clip",
    "hint": "Sutherland-Hodgman: iterate over each edge of the clip polygon and clip the subject against that half-plane. 'Inside' = to the left of the directed edge (cross product >= 0). Four cases per subject edge (S→E): both inside → output E; S inside E outside → output intersection; S outside E inside → output intersection then E; both outside → output nothing.",
    "tests": [
        {
            "name": "Square clipped by a smaller square — intersection is the smaller square",
            "code": """
import numpy as np
big  = np.array([[0.0,0.0],[4.0,0.0],[4.0,4.0],[0.0,4.0]])
clip = np.array([[1.0,1.0],[3.0,1.0],[3.0,3.0],[1.0,3.0]])
result = {fn}(big, clip)
result = np.array(result, dtype=float)
def area(p):
    x,y=p[:,0],p[:,1]; return 0.5*abs(np.sum(x*np.roll(y,-1)-np.roll(x,-1)*y))
assert len(result) >= 3, f"Result has {len(result)} points"
assert abs(area(result) - 4.0) < 1e-6, f"Intersection area should be 4.0, got {area(result):.6f}"
""",
        },
        {
            "name": "Subject fully inside clip → output equals subject",
            "code": """
import numpy as np
subject = np.array([[1.0,1.0],[2.0,1.0],[2.0,2.0],[1.0,2.0]])
clip    = np.array([[0.0,0.0],[4.0,0.0],[4.0,4.0],[0.0,4.0]])
result = {fn}(subject, clip)
result = np.array(result, dtype=float)
def area(p):
    x,y=p[:,0],p[:,1]; return 0.5*abs(np.sum(x*np.roll(y,-1)-np.roll(x,-1)*y))
assert abs(area(result) - 1.0) < 1e-6, f"Fully inside: area should be 1.0, got {area(result):.6f}"
""",
        },
        {
            "name": "Subject fully outside clip → empty output",
            "code": """
import numpy as np
subject = np.array([[10.0,10.0],[12.0,10.0],[12.0,12.0],[10.0,12.0]])
clip    = np.array([[0.0,0.0],[4.0,0.0],[4.0,4.0],[0.0,4.0]])
result = {fn}(subject, clip)
result = np.array(result)
assert len(result) == 0, f"No intersection expected, got {len(result)} points"
""",
        },
        {
            "name": "Partial overlap — triangular clip of a square",
            "code": """
import numpy as np
# A 2x2 square clipped by a right triangle occupying half of it
square = np.array([[0.0,0.0],[2.0,0.0],[2.0,2.0],[0.0,2.0]])
# Triangle clip: lower-left half (below the diagonal y=x)
clip   = np.array([[0.0,0.0],[2.0,0.0],[0.0,2.0]])
result = {fn}(square, clip)
result = np.array(result, dtype=float)
def area(p):
    x,y=p[:,0],p[:,1]; return 0.5*abs(np.sum(x*np.roll(y,-1)-np.roll(x,-1)*y))
assert len(result) >= 3
assert abs(area(result) - 2.0) < 1e-6, f"Triangle area should be 2.0, got {area(result):.6f}"
""",
        },
        {
            "name": "Large: subject with 80 vertices clipped by 20-sided polygon",
            "code": """
import numpy as np, time
n_sub, n_clip = 80, 20
angles_s = np.linspace(0, 2*np.pi, n_sub,  endpoint=False)
angles_c = np.linspace(0, 2*np.pi, n_clip, endpoint=False)
subject = np.stack([2*np.cos(angles_s), 2*np.sin(angles_s)], axis=1)
clip    = np.stack([1.5*np.cos(angles_c), 1.5*np.sin(angles_c)], axis=1)
t0 = time.time()
result = {fn}(subject, clip)
elapsed = time.time() - t0
result = np.array(result, dtype=float)
assert len(result) >= 3, "Should have intersection"
assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s"
""",
        },
    ],
}
