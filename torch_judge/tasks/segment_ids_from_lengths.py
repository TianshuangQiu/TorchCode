"""Segment IDs from Lengths task."""

TASK = {
    "title": "Segment IDs from Lengths",
    "difficulty": "Easy",
    "function_name": "segment_ids_from_lengths",
    "hint": "Two clean approaches: (1) torch.repeat_interleave(arange(B), lengths) — one line. (2) Build a zero tensor of shape (total,), place 1s at cumsum boundary positions, then cumsum the whole thing. The primitive is repeat_interleave or cumsum — not a loop.",
    "tests": [
        {
            "name": "Basic example [3,1,4,2]",
            "code": """
import torch
lengths = torch.tensor([3, 1, 4, 2])
result = {fn}(lengths)
expected = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 3, 3])
assert result.shape == expected.shape, f'Shape mismatch: {result.shape} vs {expected.shape}'
assert torch.equal(result, expected), f'Got {result.tolist()}, expected {expected.tolist()}'
""",
        },
        {
            "name": "Single segment",
            "code": """
import torch
result = {fn}(torch.tensor([5]))
expected = torch.zeros(5, dtype=torch.long)
assert torch.equal(result, expected), f'Got {result.tolist()}'
""",
        },
        {
            "name": "Zero-length segments [0,3,0,1]",
            "code": """
import torch
lengths = torch.tensor([0, 3, 0, 1])
result = {fn}(lengths)
expected = torch.tensor([1, 1, 1, 3])
assert result.shape == expected.shape, f'Shape: {result.shape}'
assert torch.equal(result, expected), f'Got {result.tolist()}, expected {expected.tolist()}'
""",
        },
        {
            "name": "All-zero lengths → empty tensor",
            "code": """
import torch
result = {fn}(torch.tensor([0, 0, 0]))
assert result.shape == (0,), f'Expected empty tensor, got shape {result.shape}'
""",
        },
        {
            "name": "Large input — 1000 segments of length 1",
            "code": """
import torch
lengths = torch.ones(1000, dtype=torch.long)
result = {fn}(lengths)
expected = torch.arange(1000)
assert result.shape == (1000,), f'Shape: {result.shape}'
assert torch.equal(result, expected), f'Large test failed at index {(result != expected).nonzero()[0].item()}'
""",
        },
        {
            "name": "Large input — 10k tokens packed into 100 segments",
            "code": """
import torch
torch.manual_seed(0)
lengths = torch.randint(1, 200, (100,))
lengths = (lengths * (10000 / lengths.sum().float())).long().clamp(min=1)
result = {fn}(lengths)
total = lengths.sum().item()
assert result.shape[0] == lengths.sum().item(), f'Total length mismatch: {result.shape[0]} vs {total}'
assert result[0].item() == 0, 'First element should be 0'
assert (result >= 0).all() and (result < 100).all(), 'IDs out of range'
""",
        },
    ],
}
