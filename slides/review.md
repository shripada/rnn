# Review of `self-attention.html`

Reviewed through the lens of Andrej Karpathy's public teaching style: derive the mechanism from simple code and numbers, track tensor shapes, and distinguish the toy model from the real implementation.

## Findings

### 1. High: the explanation of why multi-head attention is needed is too strong

The deck says one head “can look for only one kind of thing at a time” ([self-attention.html](self-attention.html#L802)). A head can attend to multiple tokens and encode several features. Its actual limitation is that it produces one attention distribution and one resulting mixture per query. It cannot preserve multiple independently weighted retrievals as cleanly as separate heads.

Suggested wording:

> One head produces one attention distribution and one mixture per word. Multiple heads can retrieve different relationships independently and preserve them in separate subspaces.

### 2. High: the residual-connection dimension claim is incorrect in the general case

“To add them… dᵥ = d” ([self-attention.html](self-attention.html#L874)) is true for the single-head toy, but not generally. In multi-head attention, each head commonly has `dᵥ = d_model / H`; concatenation followed by `W_O` returns the result to `d_model`.

Suggested wording:

> The attention block's final output must have length d. In the toy, dᵥ = d; with multiple heads, concatenation and W_O restore length d.

### 3. Medium: the complexity slide compares incomplete costs and draws too broad a conclusion

The `T²d` entry ([self-attention.html](self-attention.html#L955)) describes attention mixing, but a practical attention layer also pays roughly `Td²` for projections. Therefore, “self-attention is cheaper when T < d” ([self-attention.html](self-attention.html#L960)) can leave learners with the wrong model of a full Transformer block.

Label the row “core sequence-mixing work,” or show:

- RNN: `O(Td²)`, sequential depth `T`
- Attention: `O(Td² + T²d)`, sequential depth `1`

The major win is parallelism and short information paths, not universally fewer operations.

### 4. Medium: the deck completes the math but never closes the loop with minimal executable code

The matrix equation is excellent ([self-attention.html](self-attention.html#L714)), but the only code handoff appears near the end and points to a larger model ([self-attention.html](self-attention.html#L977)). A Karpathy-style treatment would immediately translate the equation into approximately ten lines:

```python
q = x @ Wq
k = x @ Wk
v = x @ Wv
wei = q @ k.transpose(-2, -1) / k.shape[-1]**0.5
wei = wei.masked_fill(mask == 0, float("-inf"))
wei = wei.softmax(dim=-1)
out = wei @ v
```

This would connect every preceding slide to what learners will actually implement.

### 5. Medium: “bag of words” is imprecise

The visible slide says self-attention sees a bag of words ([self-attention.html](self-attention.html#L754)), while the notes correctly call it permutation equivariant. A bag-of-words representation is normally permutation invariant; position-free self-attention preserves token correspondence and simply permutes its outputs with its inputs.

Suggested wording:

> Without positions, self-attention cannot distinguish order: permuting the inputs merely permutes the outputs.

### 6. Low: several slides are textbook pages rather than presentation slides

The rendered deck is polished and legible, but slides such as “Two problems,” the FFN diagram, and the runnable-experiments table carry a lot of prose. At projector distance, 22–24 px copy and full-width paragraphs will compete with the diagrams. The 42-slide length also makes the central mechanism feel farther away than necessary.

Target roughly 25–30 main slides and move the extended arithmetic, history, and interview questions into an appendix.

## What works especially well

- The progression from naïve `x·x` attention to Q, K, and V is excellent. Learners see the failure before receiving the abstraction.
- The three-dimensional semantic toy is readable and reused consistently.
- The order-blindness experiment is much stronger than merely stating that positional information is required.
- The two-layer pronoun example gives a real reason for depth rather than saying “deeper is better.”
- Visual semantics are consistent, the dark theme renders cleanly, and illustrative versus trained weights are disclosed carefully.
- Speaker notes are unusually thoughtful and often more technically precise than the visible slides.

## Overall verdict

This is strong courseware and substantially better than the usual “here is the QKV formula” explanation. Its main weakness is that it over-explains the arithmetic while under-emphasizing the compact implementation. Tighten the three technical claims above, insert a minimal attention implementation immediately after the matrix equation, and move about a quarter of the material to an appendix. That would make the deck feel much more direct, code-grounded, and memorable.
