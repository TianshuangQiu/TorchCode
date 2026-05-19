"""Polygon Area (Shoelace) task."""

TASK = {
    "title": "Polygon Area (Shoelace)",
    "difficulty": "Easy",
    "function_name": "polygon_area",
    "hint": "Use np.roll to shift vertices by one position: area = 0.5 * abs(sum(x * roll(y,-1) - roll(x,-1) * y)). This is the shoelace formula — no for-loop needed.",
    "tests": [
        {
            "name": "Unit square → 1.0",
            "code": """
import numpy as np
square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
result = {fn}(square)
assert abs(result - 1.0) < 1e-9, f"Unit square area: {result}"
""",
        },
        {
            "name": "Right triangle → 0.5 * base * height",
            "code": """
import numpy as np
tri = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
result = {fn}(tri)
assert abs(result - 6.0) < 1e-9, f"Triangle area: {result}, expected 6.0"
""",
        },
        {
            "name": "CW and CCW orderings give the same absolute area",
            "code": """
import numpy as np
ccw = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]])
cw  = ccw[::-1].copy()
area_ccw = {fn}(ccw)
area_cw  = {fn}(cw)
assert abs(area_ccw - 6.0) < 1e-9, f"CCW area: {area_ccw}"
assert abs(area_cw  - 6.0) < 1e-9, f"CW area:  {area_cw}"
""",
        },
        {
            "name": "Regular hexagon with unit circumradius",
            "code": """
import numpy as np
n = 6
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
hex_pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
result = {fn}(hex_pts)
expected = 3 * np.sqrt(3) / 2
assert abs(result - expected) < 1e-9, f"Hexagon area: {result}, expected {expected:.6f}"
""",
        },
        {
            "name": "Large V=100000 polygon approximating a unit circle",
            "code": """
import numpy as np, time
n = 100_000
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
t0 = time.time()
result = {fn}(poly)
elapsed = time.time() - t0
assert abs(result - np.pi) < 0.001, f"Circle approx area: {result}, expected ~pi"
assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s (expected <1s — use np.roll, not a loop)"
""",
        },
    ],
}
