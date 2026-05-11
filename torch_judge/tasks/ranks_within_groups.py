"""Ranks Within Groups task."""

TASK = {
    "title": "Ranks Within Groups",
    "difficulty": "Medium",
    "function_name": "ranks_within_groups",
    "hint": "Map group_ids to contiguous indices with unique(return_inverse=True). Build a one-hot matrix (N, G) via scatter_, then cumsum(dim=0) gives a running count per group. gather at each row's own group column, subtract 1 for 0-indexed ranks.",
    "tests": [
        {
            "name": "Spec example [2,0,2,1,0,2]",
            "code": """
import torch
group_ids = torch.tensor([2, 0, 2, 1, 0, 2])
result = {fn}(group_ids)
expected = torch.tensor([0, 0, 1, 0, 1, 2])
assert result.shape == expected.shape, f'Shape: {result.shape}'
assert torch.equal(result, expected), f'Got {result.tolist()}, expected {expected.tolist()}'
""",
        },
        {
            "name": "All same group → ascending ranks",
            "code": """
import torch
group_ids = torch.tensor([3, 3, 3, 3])
result = {fn}(group_ids)
expected = torch.tensor([0, 1, 2, 3])
assert torch.equal(result, expected), f'Got {result.tolist()}'
""",
        },
        {
            "name": "All distinct groups → all rank 0",
            "code": """
import torch
group_ids = torch.tensor([5, 2, 7, 1])
result = {fn}(group_ids)
expected = torch.zeros(4, dtype=torch.long)
assert torch.equal(result, expected), f'Got {result.tolist()}'
""",
        },
        {
            "name": "Non-contiguous group IDs [100, 100, 50]",
            "code": """
import torch
group_ids = torch.tensor([100, 100, 50])
result = {fn}(group_ids)
expected = torch.tensor([0, 1, 0])
assert torch.equal(result, expected), f'Got {result.tolist()}'
""",
        },
        {
            "name": "Ranks are 0-indexed and non-negative",
            "code": """
import torch
torch.manual_seed(42)
group_ids = torch.randint(0, 5, (20,))
result = {fn}(group_ids)
assert result.shape == (20,), f'Shape: {result.shape}'
assert (result >= 0).all(), f'Negative ranks found: {result}'
""",
        },
        {
            "name": "Large input N=10000 random groups in [0, 100)",
            "code": """
import torch, time
torch.manual_seed(0)
N = 10000
group_ids = torch.randint(0, 100, (N,))
t0 = time.time()
result = {fn}(group_ids)
elapsed = time.time() - t0
assert result.shape == (N,), f'Shape: {result.shape}'
assert (result >= 0).all(), 'Negative rank'
# verify a few manually: count how many times each id appeared before position i
for i in [0, 1, 100, 999]:
    gid = group_ids[i].item()
    expected_rank = (group_ids[:i] == gid).sum().item()
    assert result[i].item() == expected_rank, f'Rank at {i}: got {result[i].item()}, expected {expected_rank}'
assert elapsed < 2.0, f'Too slow: {elapsed:.2f}s (expected <2s — no loops)'
""",
        },
    ],
}
