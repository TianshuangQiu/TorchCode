"""Masked Pairwise Squared Distance Within Groups task."""

TASK = {
    "title": "Masked Pairwise Distance",
    "difficulty": "Medium",
    "function_name": "masked_pairwise_dist",
    "hint": "Two outer-product operations give you everything. (1) Squared distance: (points[:,None,:] - points[None,:,:]).pow(2).sum(-1) gives shape (N,N). (2) Group mask: group_ids[:,None] == group_ids[None,:] gives (N,N) bool. Combine with masked_fill(~mask, float('inf')).",
    "tests": [
        {
            "name": "Spec example (4 points, 2 groups)",
            "code": """
import torch, math
points    = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[10.,10.]])
group_ids = torch.tensor([0, 0, 1, 0])
D = {fn}(points, group_ids)
assert D.shape == (4, 4), f'Shape: {D.shape}'
assert D[0,1].item() == 1.0,  f'D[0,1]={D[0,1]}'
assert D[1,3].item() == 181.0, f'D[1,3]={D[1,3]}'
assert D[0,3].item() == 200.0, f'D[0,3]={D[0,3]}'
assert math.isinf(D[0,2].item()), f'D[0,2] should be inf, got {D[0,2]}'
assert math.isinf(D[2,0].item()), f'D[2,0] should be inf, got {D[2,0]}'
assert math.isinf(D[2,1].item()), f'D[2,1] should be inf, got {D[2,1]}'
assert math.isinf(D[2,3].item()), f'D[2,3] should be inf, got {D[2,3]}'
""",
        },
        {
            "name": "Diagonal is always 0",
            "code": """
import torch
torch.manual_seed(42)
points = torch.randn(8, 4)
group_ids = torch.randint(0, 3, (8,))
D = {fn}(points, group_ids)
diag = D.diagonal()
assert (diag == 0).all(), f'Diagonal not all zero: {diag}'
""",
        },
        {
            "name": "Output is symmetric",
            "code": """
import torch
torch.manual_seed(7)
points = torch.randn(10, 6)
group_ids = torch.randint(0, 4, (10,))
D = {fn}(points, group_ids)
assert torch.allclose(D, D.T, equal_nan=False), 'Output is not symmetric'
""",
        },
        {
            "name": "All same group → no inf",
            "code": """
import torch, math
points = torch.randn(5, 3)
group_ids = torch.zeros(5, dtype=torch.long)
D = {fn}(points, group_ids)
assert not any(math.isinf(D[i,j].item()) for i in range(5) for j in range(5)), 'Should have no inf when all same group'
""",
        },
        {
            "name": "All distinct groups → only diagonal is finite",
            "code": """
import torch, math
points = torch.randn(4, 2)
group_ids = torch.tensor([0, 1, 2, 3])
D = {fn}(points, group_ids)
for i in range(4):
    for j in range(4):
        if i == j:
            assert D[i,j].item() == 0.0, f'Diagonal D[{i},{j}] should be 0'
        else:
            assert math.isinf(D[i,j].item()), f'Off-diagonal D[{i},{j}] should be inf'
""",
        },
        {
            "name": "Large input N=500 D=64 — must complete quickly",
            "code": """
import torch, time
torch.manual_seed(0)
N, D = 500, 64
points = torch.randn(N, D)
group_ids = torch.randint(0, 10, (N,))
t0 = time.time()
result = {fn}(points, group_ids)
elapsed = time.time() - t0
assert result.shape == (N, N), f'Shape: {result.shape}'
assert elapsed < 2.0, f'Too slow: {elapsed:.2f}s (expected <2s — no loops)'
""",
        },
    ],
}
