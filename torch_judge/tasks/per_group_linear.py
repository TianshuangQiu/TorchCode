"""Per-Group Linear Layer (MoE Forward) task."""

TASK = {
    "title": "Per-Group Linear (MoE Forward)",
    "difficulty": "Hard",
    "function_name": "per_group_linear",
    "hint": "Use group_ids as a fancy index into W and b: W[group_ids] has shape (N, d_out, d_in). Then it becomes a batched matrix-vector multiply: torch.bmm(W[group_ids], x.unsqueeze(-1)).squeeze(-1) + b[group_ids]. Alternatively: torch.einsum('noi,ni->no', W[group_ids], x) + b[group_ids].",
    "tests": [
        {
            "name": "Spec example (N=3, G=2, d_in=2, d_out=2)",
            "code": """
import torch
x         = torch.tensor([[1.,0.],[0.,1.],[1.,1.]])   # (3, 2)
group_ids = torch.tensor([0, 1, 0])
W = torch.zeros(2, 2, 2)
W[0] = torch.eye(2)          # identity
W[1] = 2 * torch.eye(2)      # 2x identity
b = torch.zeros(2, 2)
b[1] = torch.tensor([1., 1.])
result = {fn}(x, group_ids, W, b)
expected = torch.tensor([[1.,0.],[1.,3.],[1.,1.]])
assert result.shape == (3, 2), f'Shape: {result.shape}'
assert torch.allclose(result, expected), f'Got {result}, expected {expected}'
""",
        },
        {
            "name": "Single expert (G=1) equals plain linear",
            "code": """
import torch
torch.manual_seed(42)
N, d_in, d_out = 8, 4, 6
x = torch.randn(N, d_in)
W = torch.randn(1, d_out, d_in)
b = torch.randn(1, d_out)
group_ids = torch.zeros(N, dtype=torch.long)
result = {fn}(x, group_ids, W, b)
expected = (x @ W[0].T) + b[0]
assert result.shape == (N, d_out), f'Shape: {result.shape}'
assert torch.allclose(result, expected, atol=1e-5), f'Mismatch vs plain linear'
""",
        },
        {
            "name": "One token per group (N=G)",
            "code": """
import torch
torch.manual_seed(7)
G, d_in, d_out = 5, 3, 4
x = torch.randn(G, d_in)
W = torch.randn(G, d_out, d_in)
b = torch.randn(G, d_out)
group_ids = torch.arange(G)
result = {fn}(x, group_ids, W, b)
assert result.shape == (G, d_out), f'Shape: {result.shape}'
for i in range(G):
    expected_i = W[i] @ x[i] + b[i]
    assert torch.allclose(result[i], expected_i, atol=1e-5), f'Token {i} mismatch'
""",
        },
        {
            "name": "Gradients flow to W and b",
            "code": """
import torch
torch.manual_seed(0)
N, G, d_in, d_out = 4, 2, 3, 3
x = torch.randn(N, d_in)
W = torch.randn(G, d_out, d_in, requires_grad=True)
b = torch.randn(G, d_out, requires_grad=True)
group_ids = torch.tensor([0, 1, 0, 1])
out = {fn}(x, group_ids, W, b)
out.sum().backward()
assert W.grad is not None, 'W.grad is None'
assert b.grad is not None, 'b.grad is None'
assert W.grad.shape == W.shape, f'W.grad shape: {W.grad.shape}'
assert b.grad.shape == b.shape, f'b.grad shape: {b.grad.shape}'
""",
        },
        {
            "name": "Large input N=1000 G=8 d_in=128 d_out=256",
            "code": """
import torch, time
torch.manual_seed(0)
N, G, d_in, d_out = 1000, 8, 128, 256
x = torch.randn(N, d_in)
W = torch.randn(G, d_out, d_in)
b = torch.randn(G, d_out)
group_ids = torch.randint(0, G, (N,))
t0 = time.time()
result = {fn}(x, group_ids, W, b)
elapsed = time.time() - t0
assert result.shape == (N, d_out), f'Shape: {result.shape}'
assert elapsed < 2.0, f'Too slow: {elapsed:.2f}s (expected <2s — no loops)'
""",
        },
    ],
}
