"""Top-k Gather task."""

TASK = {
    "title": "Top-k Gather",
    "difficulty": "Medium",
    "function_name": "topk_gather",
    "hint": "torch.topk(scores, k, dim=-1) returns indices of shape (B, k). To gather from values of shape (B, N, D) you need an index tensor of the same shape: unsqueeze(-1) on the (B, k) indices, then expand(-1, -1, D), then gather(dim=1, ...).",
    "tests": [
        {
            "name": "Basic correctness (1, 4, 2) k=2",
            "code": """
import torch
scores = torch.tensor([[0.1, 0.9, 0.4, 0.7]])          # (1, 4)
values = torch.tensor([[[1,1],[2,2],[3,3],[4,4]]]).float()  # (1, 4, 2)
result = {fn}(scores, values, k=2)
assert result.shape == (1, 2, 2), f'Shape: {result.shape}'
# top-2 scores are 0.9 (idx=1) and 0.7 (idx=3) → values [2,2] and [4,4]
assert torch.allclose(result[0, 0], torch.tensor([2., 2.])), f'First: {result[0,0]}'
assert torch.allclose(result[0, 1], torch.tensor([4., 4.])), f'Second: {result[0,1]}'
""",
        },
        {
            "name": "k=1 preserves all dimensions",
            "code": """
import torch
torch.manual_seed(42)
B, N, D = 4, 10, 8
scores = torch.randn(B, N)
values = torch.randn(B, N, D)
result = {fn}(scores, values, k=1)
assert result.shape == (B, 1, D), f'Shape with k=1: {result.shape}, expected ({B}, 1, {D})'
""",
        },
        {
            "name": "k=N returns all values ordered by score",
            "code": """
import torch
torch.manual_seed(7)
B, N, D = 2, 5, 4
scores = torch.randn(B, N)
values = torch.randn(B, N, D)
result = {fn}(scores, values, k=N)
assert result.shape == (B, N, D), f'Shape with k=N: {result.shape}'
# each row should be values reordered by descending score
for b in range(B):
    order = scores[b].argsort(descending=True)
    expected_row = values[b][order]
    assert torch.allclose(result[b], expected_row), f'Order wrong for batch {b}'
""",
        },
        {
            "name": "Batch of 2, values shape (2, 5, 3) k=2",
            "code": """
import torch
torch.manual_seed(99)
scores = torch.randn(2, 5)
values = torch.randn(2, 5, 3)
result = {fn}(scores, values, k=2)
assert result.shape == (2, 2, 3), f'Shape: {result.shape}'
# verify values match topk indices
topk_idx = scores.topk(2, dim=-1).indices  # (2, 2)
for b in range(2):
    for ki in range(2):
        expected = values[b, topk_idx[b, ki]]
        assert torch.allclose(result[b, ki], expected), f'Mismatch b={b} ki={ki}'
""",
        },
        {
            "name": "Large input (32, 1000, 64) k=10",
            "code": """
import torch
torch.manual_seed(0)
scores = torch.randn(32, 1000)
values = torch.randn(32, 1000, 64)
result = {fn}(scores, values, k=10)
assert result.shape == (32, 10, 64), f'Shape: {result.shape}'
""",
        },
    ],
}
