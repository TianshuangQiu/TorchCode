# TorchCode — Context for Claude

## User Profile

Preparing for ML engineering interviews. Strong theoretical foundation (einsum, attention, transformer internals) but specific gaps in tensor manipulation under pressure and algorithm breadth. Recently failed an interview question on block attention masks for packed sequences despite having studied most of TorchCode.

---

## Known Weaknesses

### Tensor manipulation

- **Loops instead of broadcasting.** Reaches for Python loops with index variables when a vectorized primitive exists. In the failed interview: mixed up `enumerate` indices with actual token positions in a packed-sequence problem.
- **Doesn't identify the primitive first.** When the right answer is a broadcasted outer-product comparison, writes nested loops instead.
- **Weaker on:** `gather`/`scatter`, `cumsum` tricks, `repeat_interleave`, fancy indexing into weight tensors (MoE-style).
- **Stronger on:** `einsum`, basic broadcasting in attention.
- **Key missing pattern:** "constraint between two sets of tokens" → outer-product comparison + mask. This is the single highest-leverage gap.

### Algorithms / LeetCode

- Very limited experience. Prior to current prep: Two Sum, LRU cache only.
- Completed: Valid Parens (LC 20), Longest Substring Without Repeating (LC 3), Level Order Traversal (LC 102), Two Sum II (LC 167), 3Sum (LC 15).
- Patterns in progress: sliding window, BFS/DFS, two pointers with duplicate handling, stack-based matching.
- Not yet practiced: DP, binary search on answer, monotonic stack, heap, graph problems beyond BFS, interval problems.

### Interview behavior

- Starts coding before narrating the approach — proximate cause of the index-variable bug.
- Coaching emphasis: state the reduction in one sentence (e.g. "I need `[i,j]` true iff `doc[i]==doc[j]` AND `chunk[i]>=chunk[j]`") before writing any code.
- Hasn't done mock interviews; solo practice doesn't replicate the cognitive load of being watched.

---

## Strengths to Leverage

- Strong conceptual grasp of attention, transformers, einsum.
- Comfortable with PyTorch surface area (modules, autograd, basic ops).
- Disciplined enough to grind problem sets.
- Takes feedback well; follows up with concrete asks.

---

## Active Workstreams

- **TorchCode extension** — added 5 tensor manipulation problems (#41–#45): segment IDs, top-k gather, masked pairwise distance, ranks within groups, per-group linear (MoE forward). Specs written and implemented.
- **LeetCode foundations** — finished 5 starter problems, ready for next batch.

---

## High-Leverage Next Steps

### Tensor manipulation

1. Drill the next tier after #41–#45: `segment_sum`/`segment_mean` (graph nets, packed losses), causal masking with chunked/grouped variants, KV-cache-style index updates.
2. Target existing TorchCode problems that compose these primitives under realistic shapes: **#9** (causal), **#10** (GQA), **#14** (KV cache), **#25** (Flash Attention).
3. On every new problem: write one sentence describing `output[i,j]` in terms of inputs before writing any code. If that sentence can't be written, the reduction isn't done yet.

### Algorithms

- **Next batch (5 problems):** DP basics — Climbing Stairs (LC 70), House Robber (LC 198); binary search — Binary Search (LC 704), Search Insert Position (LC 35); sliding window — Minimum Size Subarray Sum (LC 209).
- **After that:** monotonic stack — Daily Temperatures (LC 739); heap — Kth Largest (LC 215); interval merging — Merge Intervals (LC 56).
- Avoid Hards until the pattern library covers ~8 categories.

### Habit drills

- **Talk-aloud practice.** Force narration of the reduction step before coding, even when practicing alone.
- **Mock interviews.** Pramp or interviewing.io, weekly. Ask interviewer to interrupt when an index variable's meaning is unclear.
- **Post-mortem framework.** After every wrong solution, classify: (a) wrong primitive, (b) right primitive wrong shape, (c) right code wrong edge case, (d) ran out of time. The mix diagnoses what to drill next.

---

## Coaching Style

- Wants direct, technical feedback. Asks for the next problem, not reassurance.
- Responds to "honest take" framing that names the actual gap.
- Prefers concrete references (LeetCode numbers, repo links). Will not act on "do more practice."
- Works in batches of ~5. Don't give 50 problems at once.
- Asks meta-questions ("how should I prepare better?") — take them seriously and give a strategy answer.

### Anti-patterns to avoid

- Don't say "grind more TorchCode" — they already did and it missed the gap. The gap is manipulation of structured data and narrating before coding, not volume.
- Don't give the full answer to a problem they're working through. Hints and pattern-naming only, unless they post code and ask for review.
- Don't pad with general advice.

---

## Generating New TorchCode Problems

### What makes a good problem for this user

A good problem for this user:

1. **Targets a specific primitive** from the gap list: `gather`, `scatter_`, `cumsum`, `repeat_interleave`, fancy indexing, outer-product comparison, `masked_fill`, `bmm`.
2. **Has a clean one-sentence reduction** — `output[i]` or `output[i,j]` should be expressible in terms of inputs without needing a loop.
3. **Is broken by a naive loop solution** — include a large-N test (N ≥ 10k) where a loop would be visibly slow.
4. **Names the primitive in the hint** — not just "think about broadcasting" but "this is an outer-product comparison; state `mask[i,j]` in terms of inputs before coding."
5. **Includes the Reduction step box** in the template — the single most important pedagogical element for this user.

### Problem categories still uncovered (prioritised)

| Priority | Category | Primitives | Example problems |
|----------|----------|------------|-----------------|
| High | Segment aggregation | `scatter_add`, `index_add` | `segment_sum(values, seg_ids)`, `segment_mean` |
| High | Variable-length masking | cumsum + broadcasting | Build a padding mask from lengths; apply it |
| Medium | Sparse index ops | `index_select`, `nonzero` | Select rows matching a condition without a loop |
| Medium | Grouped normalization | fancy indexing + broadcasting | Normalize each token by its group's mean/std |
| Low | KV-cache index updates | `index_copy_`, `scatter` | Append to a pre-allocated cache at dynamic positions |

### Difficulty calibration

| Difficulty | Characteristic |
|------------|---------------|
| Easy | Single primitive, ≤2 tensor ops, 1D or simple 2D shapes |
| Medium | 2–3 primitives, requires reasoning about broadcast dimensions explicitly |
| Hard | 3+ primitives, gradient flow required, or non-obvious index arithmetic |

---

## TorchCode Repo: Adding New Problems

### Quick scaffold

```bash
python scripts/new_problem.py <number> <task_id> "<Title>" <Easy|Medium|Hard>
# Example:
python scripts/new_problem.py 46 segment_sum "Segment Sum" Medium
```

Creates all three files and prints the README row. Then fill in the TODOs.

### File locations

| File | Purpose |
|------|---------|
| `torch_judge/tasks/<task_id>.py` | Task definition — auto-discovered, no registration needed |
| `templates/NN_<task_id>.ipynb` | Blank notebook for the user |
| `solutions/NN_<task_id>_solution.ipynb` | Reference solution |

### Task file format

```python
TASK = {
    "title": "Human-readable title",
    "difficulty": "Easy",  # "Easy" | "Medium" | "Hard"
    "function_name": "snake_case_fn_name",  # must match what the user defines
    "hint": "Name the primitive. E.g. 'use repeat_interleave or cumsum — not a loop'.",
    "tests": [
        {
            "name": "Short description of what this test checks",
            "code": """
import torch
result = {fn}(torch.tensor([3, 1, 4]))   # {fn} is replaced with function_name at runtime
assert result.shape == (8,), f'Shape: {result.shape}'
assert torch.equal(result, torch.tensor([0, 0, 0, 1, 2, 2, 2, 2]))
""",
        },
    ],
}
```

**Test rules:**
- `{fn}` is replaced with `function_name` at runtime — use it everywhere, never hardcode the name
- `AssertionError` = test failed; any other exception = crash (both shown as failures)
- Include at minimum: basic correctness, one edge case, one large-N test (N ≥ 10k)
- Use `torch.equal` for exact integer matches; `torch.allclose(atol=1e-5)` for floats
- Time the large-N test with `time.time()` and assert `elapsed < 2.0` to penalise loops

### Notebook format (6 cells, fixed order)

| Cell | Type | Content |
|------|------|---------|
| 0 | Markdown | Colab badge + `# emoji Difficulty: Title` + description + signature + rules + example + **Reduction step** box |
| 1 | Code | Colab pip installer (`try: import google.colab ...`) — copy verbatim from any existing notebook |
| 2 | Code | `import torch` |
| 3 | Code | Skeleton `def fn(...): pass` (template) or complete solution (solution notebook) |
| 4 | Code | Debug prints / verification |
| 5 | Code | `from torch_judge import check; check("task_id")` |

**Reduction step box** — required in every template cell 0:

```
> **Reduction step (say this before coding):** `output[i]` is ... (one sentence relating each
> output element to input elements, naming the structured access pattern)
```

This is the most important pedagogical element for this user. Always include it.

### Solution notebook conventions

- Cell 0: `# emoji Solution: Title` + `**Primitive:** name the core op` + `**Reduction:** one sentence`
- Cell 3: first line is `# primitive: repeat_interleave` (or whichever applies)
- Show two approaches when two clean vectorised solutions exist (e.g. `repeat_interleave` vs `cumsum` boundary trick)

### README update checklist

1. Update badge count: `problems-N-orange` → new count (line ~30 of README.md)
2. Update "N curated problems" in the features table (line ~47)
3. Paste the README row into the correct section table
4. Row format:

```
| N | <a href="GITHUB_URL">Title</a> <a href="COLAB_URL"><img src="...colab-badge.svg" height="20"></a> | `fn(args)` | ![Difficulty badge] | ⭐ | key concepts |
```

**Known issue — always verify README edits with grep:**
The Edit tool has been observed to report "updated successfully" for large README insertions without actually writing the content. After any README edit, confirm with:

```bash
grep -n "search term" README.md
```

If the content is missing, re-run the edit using a unique anchor string (e.g. the full Conv2d table row) rather than just the section heading.
