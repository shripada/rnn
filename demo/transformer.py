# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "numpy"]
# ///
"""
==============================================================================
 The date translator again, as a Transformer: no RNN anywhere
==============================================================================

THE PROBLEM
-----------
The same task as seq2seq.py (Part 3) and attention.py (Part 4): read a date
written any of six ways, write it as YYYY-MM-DD. Same data, same exam.

    "23 Oct 2026"                                     -> "2026-10-23"
    "the twenty-third of October, twenty twenty-six"  -> "2026-10-23"

THE MODEL (Vaswani et al., 2017, "Attention Is All You Need")
-------------------------------------------------------------
    ENCODER  embed each character, add its position (Part 5's sine waves),
             then N identical blocks:
                 self-attention    every character looks at every character
                 feed-forward      each character on its own: d -> 4d -> d
             each wrapped as  x = LayerNorm(x + sublayer(x))

    DECODER  embed what has been written so far, add positions, then N blocks:
                 masked self-attention   look only at what's already written
                 cross-attention         queries from here, keys and values
                                         from the encoder: Part 4's look-back
                 feed-forward
             then a linear layer + softmax over the output alphabet.

Written out block by block (no nn.Transformer) so each slide points at a few
named lines. Default size: d = 64, 4 heads, 2 encoder + 2 decoder blocks.

TRAINING vs WRITING
-------------------
  Training:  the whole right answer goes in at once, shifted right (teacher
             forcing, Part 3). The mask stops position t peeking at t+1, so
             ONE pass gives the loss at every position.
  Writing:   still one character at a time: write, feed it back, run again.

THE COMMANDS
------------
  train      build and train the model, save it
  test       grade it on unseen dates and unseen years (same exam as Parts 3-4)
  translate  give it a date; --details shows where cross-attention looked
  speed      time one training round: GRU + attention vs Transformer
  export     put the trained weights into slides/transformer.html (run it for
             the default model and for --max-dates 4)

  uv run demo/transformer.py train                     # ~3 min on a laptop CPU
  uv run demo/transformer.py test
  uv run demo/transformer.py translate "23 Oct 2026" --details
  uv run demo/transformer.py translate "23 Oct 2026" --details --layer 1 --head 3
  uv run demo/transformer.py speed
  uv run demo/transformer.py export

THE BOTTLENECK TEST, AGAIN
--------------------------
  uv run demo/transformer.py train --max-dates 4
  uv run demo/transformer.py test  --max-dates 4

HOW THIS MAPS TO THE SLIDES (Part 6)
------------------------------------
  embeddings + positions   Transformer.embed, positions()
  multi-head attention     MultiHeadAttention (Part 5, unchanged)
  feed-forward             FeedForward
  add & norm               EncoderBlock.forward / DecoderBlock.forward
  cross-attention          DecoderBlock.cross
  linear + softmax         Transformer.head (shares weights with tgt_embed)
  training in one pass     Transformer.forward(teacher_forcing=True)
  writing one at a time    Transformer.decode_step
"""

import argparse
import base64
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The data, alphabets, exam and greedy / beam search are exactly the ones in seq2seq.py;
# the attention map printer is attention.py's.
from attention import AttnSeq2Seq, attention_map
from seq2seq import (END, FORMATS, IN_CHARS, LENGTH_BUCKETS, OUT_CHARS, PAD, all_right,
                     beam_search, encode_in, example, grade, make_batch, max_out, split_dates,
                     target, test_many, write_date)

HERE = Path(__file__).parent
DECK = HERE.parent / "slides" / "transformer.html"


def model_path(d: int, layers: int, max_dates: int = 1) -> Path:
    """demo/transformer-64-2.pt, demo/transformer-64-2-x4.pt, ..."""
    return HERE / (f"transformer-{d}-{layers}" + (f"-x{max_dates}" if max_dates > 1 else "") + ".pt")


# =============================================================================
# 1. THE PARTS
# =============================================================================

def positions(T: int, d: int) -> torch.Tensor:
    """Part 5's sine waves: row t is position t, written as d/2 sines and d/2 cosines
    at wavelengths from 2π up to 10000·2π. (T, d)"""
    t = torch.arange(T)[:, None].float()
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2).float() / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(t * freq), torch.cos(t * freq)
    return pe


class MultiHeadAttention(nn.Module):
    """Part 5's multi-head attention, unchanged:

        Q = q_in W_q,  K = kv_in W_k,  V = kv_in W_v     split into `heads` slices of d/heads
        A = softmax(Q Kᵀ / √d_k)                          one map per head; mask -> weight 0
        out = concat(A V over heads) W_o

    Self-attention: q_in and kv_in are the same sequence.
    Cross-attention: q_in is the decoder, kv_in is the encoder's output."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.d_k = heads, d // heads
        self.W_q, self.W_k, self.W_v, self.W_o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, q_in, kv_in, mask):
        """mask: (n, 1 or Tq, Tk), True where looking is allowed.
        Returns the output (n, Tq, d) and the weights (n, heads, Tq, Tk)."""
        n, Tq, Tk = q_in.shape[0], q_in.shape[1], kv_in.shape[1]
        split = lambda t, T: t.view(n, T, self.heads, self.d_k).transpose(1, 2)   # (n, heads, T, d_k)
        Q, K, V = split(self.W_q(q_in), Tq), split(self.W_k(kv_in), Tk), split(self.W_v(kv_in), Tk)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)                     # (n, heads, Tq, Tk)
        a = scores.masked_fill(~mask[:, None], float("-inf")).softmax(-1)
        out = (a @ V).transpose(1, 2).reshape(n, Tq, -1)                            # heads side by side
        return self.W_o(out), a


class FeedForward(nn.Module):
    """The same small network applied to every position on its own:
    d -> 4d (ReLU) -> d. Attention moves information between positions; this
    is where each position works on what it has gathered."""

    def __init__(self, d: int, ff: int):
        super().__init__()
        self.up, self.down = nn.Linear(d, ff), nn.Linear(ff, d)

    def forward(self, x):
        return self.down(F.relu(self.up(x)))


class EncoderBlock(nn.Module):
    """x = LayerNorm(x + SelfAttention(x));  x = LayerNorm(x + FeedForward(x))
    The "x +" is the residual path: each sublayer ADDS to x instead of replacing it.
    (Norm after the add, as in the paper.)"""

    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.attn, self.ff = MultiHeadAttention(d, heads), FeedForward(d, ff)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, mask):
        x = self.norm1(x + self.attn(x, x, mask)[0])
        return self.norm2(x + self.ff(x))


class DecoderBlock(nn.Module):
    """x = LayerNorm(x + MaskedSelfAttention(x))         look back at what's written
       x = LayerNorm(x + CrossAttention(x, memory))       look at the input (Part 4)
       x = LayerNorm(x + FeedForward(x))"""

    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.self_attn, self.cross = MultiHeadAttention(d, heads), MultiHeadAttention(d, heads)
        self.ff = FeedForward(d, ff)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, memory, causal, src_mask):
        x = self.norm1(x + self.self_attn(x, x, causal)[0])
        c, a = self.cross(x, memory, src_mask)
        x = self.norm2(x + c)
        return self.norm3(x + self.ff(x)), a


# =============================================================================
# 2. THE WHOLE MACHINE
# =============================================================================

class Transformer(nn.Module):
    """
        ENCODER   z_0 = embed(x)·√d + positions          one row per input character
                  z_l = EncoderBlock(z_{l-1})            l = 1 .. layers
                  memory = z_layers                      every character, fully mixed
        DECODER   u_0 = embed(y_in)·√d + positions       y_in = ⟨s⟩ + the answer so far
                  u_l = DecoderBlock(u_{l-1}, memory)
                  P(next char) = softmax(u_layers · Eᵀ)  E = the output embedding, reused

    The decoder's state for decode_step is a tuple (memory, src_mask, written so far,
    last cross-attention weights), so greedy and beam search from seq2seq.py work on it."""

    def __init__(self, d: int = 64, heads: int = 4, layers: int = 2, ff: int = 256, max_dates: int = 1):
        super().__init__()
        self.d, self.heads, self.layers, self.ff, self.max_dates = d, heads, layers, ff, max_dates
        self.src_embed = nn.Embedding(len(IN_CHARS), d)
        self.tgt_embed = nn.Embedding(len(OUT_CHARS), d)
        for e in (self.src_embed, self.tgt_embed):
            nn.init.normal_(e.weight, std=d ** -0.5)       # so embed·√d starts near size 1, like the positions
        self.register_buffer("pos", positions(512, d), persistent=False)
        self.encoder = nn.ModuleList(EncoderBlock(d, heads, ff) for _ in range(layers))
        self.decoder = nn.ModuleList(DecoderBlock(d, heads, ff) for _ in range(layers))
        self.head = nn.Linear(d, len(OUT_CHARS))
        self.head.weight = self.tgt_embed.weight           # the paper shares these: score = u · e_char

    def embed(self, table, ids):
        return table(ids) * math.sqrt(self.d) + self.pos[: ids.shape[1]]

    def encode(self, x, lengths):
        """Run the encoder. Returns the decoder's starting state."""
        src_mask = (torch.arange(x.shape[1])[None, :] < lengths[:, None])[:, None, :]   # (n, 1, T): not padding
        z = self.embed(self.src_embed, x)
        for block in self.encoder:
            z = block(z, src_mask)
        return z, src_mask, x.new_zeros(x.shape[0], 0), None

    def decode(self, ys, memory, src_mask):
        """Run the decoder over everything in ys at once. Returns scores at every
        position (n, t, len(OUT_CHARS)) and each block's cross-attention weights."""
        t = ys.shape[1]
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool))[None]    # position i sees 0..i only
        u, weights = self.embed(self.tgt_embed, ys), []
        for block in self.decoder:
            u, a = block(u, memory, causal, src_mask)
            weights.append(a)
        return self.head(u), weights

    def decode_step(self, y_prev, state):
        """Write ONE character: append y_prev to what's written and run the decoder
        again over all of it. (Real systems cache the old keys and values; we don't.)"""
        memory, src_mask, ys, _ = state
        ys = torch.cat([ys, y_prev[:, None]], 1)
        logits, weights = self.decode(ys, memory, src_mask)
        a = torch.stack([w[:, :, -1] for w in weights], 1)                # (n, layers, heads, T)
        return logits[:, -1], (memory, src_mask, ys, a)

    def forward(self, x, lengths, y_in, teacher_forcing: bool = True):
        """Scores at every output step: (batch, steps, len(OUT_CHARS)), as in seq2seq.py.

        teacher_forcing=True:  ONE pass over the whole y_in. This is training.
        teacher_forcing=False: write one character at a time and feed it back,
                               exactly as when nobody knows the answer."""
        state = self.encode(x, lengths)
        if teacher_forcing:
            return self.decode(y_in, state[0], state[1])[0]
        y_prev, scores = y_in[:, 0], []
        for _ in range(y_in.shape[1]):
            logits, state = self.decode_step(y_prev, state)
            scores.append(logits)
            y_prev = logits.argmax(-1)
        return torch.stack(scores, 1)


def load_model(args):
    path = model_path(args.d, args.layers, args.max_dates)
    if not path.exists():
        raise SystemExit(f"No trained model at {path}. Run `train{flags(args)}` first.")
    ckpt = torch.load(path, weights_only=True)
    model = Transformer(ckpt["d"], ckpt["heads"], ckpt["layers"], ckpt["ff"], ckpt["max_dates"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def flags(args) -> str:
    return ((f" --d {args.d}" if args.d != 64 else "")
            + (f" --layers {args.layers}" if args.layers != 2 else "")
            + (f" --max-dates {args.max_dates}" if args.max_dates != 1 else ""))


def count(*mods) -> int:
    """Weights in these modules, counting a shared tensor once."""
    return sum(p.numel() for p in {id(p): p for m in mods for p in m.parameters()}.values())


# =============================================================================
# 3. GREEDY, keeping the cross-attention weights so we can show them
# =============================================================================

@torch.no_grad()
def greedy_with_weights(model: Transformer, text: str):
    """Like seq2seq.greedy: returns the answer, the top 3 guesses at each step, and
    the cross-attention weights at each step: (layers, heads, input length) each."""
    x = torch.tensor([encode_in(text)])
    state = model.encode(x, torch.tensor([x.shape[1]]))
    y_prev, out, steps, weights = torch.tensor([0]), "", [], []
    for _ in range(max_out(model.max_dates)):
        logits, state = model.decode_step(y_prev, state)
        weights.append(state[-1][0])
        best = logits.softmax(-1)[0].topk(3)
        steps.append([(OUT_CHARS[i], v) for v, i in zip(best.values.tolist(), best.indices.tolist())])
        y_prev = best.indices[:1]
        if OUT_CHARS[y_prev.item()] == END:
            break
        out += OUT_CHARS[y_prev.item()]
    return out, steps, weights


# =============================================================================
# 4. TRAIN: the same data as Parts 3-4, all positions in one pass
# =============================================================================

def cmd_train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train_dates, test_days, _ = split_dates()

    if args.max_dates == 1:
        print("THE PROBLEM (same as seq2seq.py): read a date written any of six ways, write it as YYYY-MM-DD.")
    else:
        print(f"THE PROBLEM (same as seq2seq.py --max-dates {args.max_dates}): read 1 to "
              f"{args.max_dates} dates, write them all.")
    for k in range(1, args.max_dates + 1):
        s, t = example(train_dates, k, rng)
        print(f"  {s:<52} -> {t}" if args.max_dates == 1 else f"  {s}\n    -> {t}")

    model = Transformer(args.d, args.heads, args.layers, args.ff, args.max_dates)
    print(f"\nModel: Transformer, d = {args.d}, {args.heads} heads, {args.layers} encoder + "
          f"{args.layers} decoder blocks, feed-forward {args.d} -> {args.ff} -> {args.d}.")
    print(f"  embeddings {count(model.src_embed, model.tgt_embed):,} weights · encoder "
          f"{count(model.encoder):,} · decoder {count(model.decoder):,} · output layer "
          f"{model.head.bias.numel():,} (+ the shared embedding) · total {count(model):,}")

    # Adam, with the learning rate rising over the first rounds (a "warm-up"): a
    # freshly built Transformer is easy to knock off course with full-size steps.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 1 - 0.9 * (s - warm) / (args.steps - warm))
    val = make_batch(test_days, 1000, random.Random(args.seed + 1), args.max_dates)

    print(f"\nTRAINING: {args.steps} rounds of {args.batch} random inputs, teacher forcing, "
          "every output position in one pass.")
    print(f"{'round':>6} {'train loss':>11} {'val loss':>9} {'val exact':>10}")
    t0, history = time.time(), []
    for step in range(1, args.steps + 1):
        x, lengths, y_in, y_out = make_batch(train_dates, args.batch, rng, args.max_dates)
        scores = model(x, lengths, y_in)
        loss = F.cross_entropy(scores.reshape(-1, len(OUT_CHARS)), y_out.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 250 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, vl, vyi, vyo = val
                vs = model(vx, vl, vyi, teacher_forcing=False)
                vloss = F.cross_entropy(vs.reshape(-1, len(OUT_CHARS)), vyo.reshape(-1), ignore_index=PAD).item()
                exact = all_right(vs, vyo).float().mean().item()
            model.train()
            history.append((step, round(time.time() - t0, 1), round(loss.item(), 4), round(vloss, 4), round(exact, 4)))
            print(f"{step:>6} {loss.item():>11.3f} {vloss:>9.3f} {exact:>10.1%}")

    print(f"\nTrained in {time.time() - t0:.0f}s.")
    torch.save({"state_dict": model.state_dict(), "d": args.d, "heads": args.heads, "layers": args.layers,
                "ff": args.ff, "max_dates": args.max_dates, "history": history},
               model_path(args.d, args.layers, args.max_dates))
    print(f"Saved the trained model to {model_path(args.d, args.layers, args.max_dates)}")
    print(f"Next: `test{flags(args)}`, or `translate \"23 Oct 2026\"{flags(args)} --details`.")


# =============================================================================
# 5. TEST: the same exam as Parts 3-4
# =============================================================================

def cmd_test(args):
    model, _ = load_model(args)
    rng = random.Random(args.seed)
    _, test_days, test_years = split_dates()
    print(f"Model: Transformer, d = {model.d}, {model.heads} heads, {model.layers} + {model.layers} "
          f"blocks, {count(model):,} weights.\n")
    if model.max_dates > 1:
        return test_many(model, test_days, rng, why="No RNN and no single c: every step looks at every date.")

    pairs, fmts = [], []
    for d in test_days:
        for f in FORMATS:
            pairs.append((write_date(d, f, rng), target(d)))
            fmts.append(f)
    right = grade(model, pairs)
    print(f"TEST: {len(pairs):,} inputs, built from {len(test_days):,} dates the model never saw.")
    print(f"Whole date exactly right: {np.mean(right):.1%}\n")
    print("By format:")
    for f in FORMATS:
        print(f"  {f:<26} {np.mean([ok for ok, ff in zip(right, fmts) if ff == f]):6.1%}")
    print("\nBy input length:")
    for lo, hi in LENGTH_BUCKETS:
        r = [ok for ok, (s, _) in zip(right, pairs) if lo <= len(s) <= hi]
        if r:
            print(f"  {lo:>2}-{hi:<2} chars  {np.mean(r):6.1%}   ({len(r):,} inputs)")

    year_pairs = [(write_date(d, f, rng), target(d)) for d in test_years for f in FORMATS]
    yr = grade(model, year_pairs)
    spelled = [ok for ok, (s, _) in zip(yr, year_pairs) if s.lower().startswith("the ")]
    digits = [ok for ok, (s, _) in zip(yr, year_pairs) if not s.lower().startswith("the ")]
    print(f"\nYears it never saw (2080-2099), all formats: {np.mean(yr):.1%}")
    print(f"  year written in digits ('2091'):             {np.mean(digits):.1%}")
    print(f"  year spelled out ('twenty ninety-one'):      {np.mean(spelled):.1%}")

    print("\nSome mistakes (input -> wrong answer, right answer):")
    wrong = [(s, t) for ok, (s, t) in zip(right + yr, pairs + year_pairs) if not ok]
    for s, t in rng.sample(wrong, min(6, len(wrong))):
        print(f"  {s:<52} -> {greedy_with_weights(model, s)[0]:<12} should be {t}")
    if not wrong:
        print("  none")


# =============================================================================
# 6. TRANSLATE: and see where cross-attention looked
# =============================================================================

def cmd_translate(args):
    model, _ = load_model(args)
    out, steps, weights = greedy_with_weights(model, args.text)
    print(f"Input: {args.text!r}  ({len(args.text)} characters)\n")
    print(f"  greedy:  {out}")
    if args.details:
        layer = (args.layer or model.layers) - 1
        if args.head:
            rows, which = [w[layer, args.head - 1].tolist() for w in weights], f"head {args.head}"
        else:
            rows, which = [w[layer].mean(0).tolist() for w in weights], f"average of the {model.heads} heads"
        print(f"\n  Cross-attention, decoder block {layer + 1}, {which} (darker = more weight):\n")
        print(attention_map(args.text, steps, rows))
    if args.beam > 1:
        print(f"\n  beam search, width {args.beam}:")
        for i, (ans, p) in enumerate(beam_search(model, args.text, args.beam), 1):
            print(f"    {i}. {ans:<14} {p:.1%}")


# =============================================================================
# 7. SPEED: one training round, RNN vs Transformer
# =============================================================================

def cmd_speed(args):
    """Time a training round (forward + backward, teacher forcing) for Part 4's
    GRU + attention and for this Transformer, on the same batches. Untrained
    weights: speed doesn't depend on what the weights are."""
    torch.manual_seed(0)
    rng = random.Random(0)
    train_dates, _, _ = split_dates()
    print(f"One training round of {args.batch} inputs, forward + backward, averaged over {args.rounds} rounds.\n")
    print(f"  {'dates per input':>15} {'avg input':>10}   {'GRU + attention':>16} {'Transformer':>12}")
    for k in (1, 4):
        batches = [make_batch(train_dates, args.batch, rng, k) for _ in range(args.rounds)]
        times = []
        for model in (AttnSeq2Seq(128, k), Transformer(max_dates=k)):
            t0 = time.time()
            for x, lengths, y_in, y_out in batches:
                loss = F.cross_entropy(model(x, lengths, y_in).reshape(-1, len(OUT_CHARS)),
                                       y_out.reshape(-1), ignore_index=PAD)
                loss.backward()
            times.append((time.time() - t0) / args.rounds * 1000)
        avg = np.mean([lengths.float().mean().item() for _, lengths, _, _ in batches])
        print(f"  {'1' if k == 1 else '1-4':>15} {avg:>8.0f} ch   {times[0]:>13.0f} ms {times[1]:>9.0f} ms")
    print("\nThe GRU takes one step per input character and one per output character, in order.")
    print("The Transformer does each block as a few big matrix multiplies over all positions at once.")


# =============================================================================
# 8. EXPORT: put the weights into the Part 6 slides
# =============================================================================

def f16(t: torch.Tensor) -> str:
    return base64.b64encode(t.detach().numpy().astype("<f2").tobytes()).decode()


def cmd_export(args):
    model, _ = load_model(args)
    blob = {"in": IN_CHARS, "out": OUT_CHARS, "d": model.d, "heads": model.heads, "layers": model.layers,
            "ff": model.ff, "maxDates": model.max_dates,
            "w": {k: {"shape": list(v.shape), "b64": f16(v)} for k, v in model.state_dict().items()
                  if k != "head.weight"}}                    # = tgt_embed.weight
    # The one-date model runs "translate a date" and "watch the heads"; the --max-dates 4
    # one runs the bottleneck slide. Each has its own slot: TF_MODEL, TF_MODEL_X4.
    name = "TF_MODEL" + (f"_X{model.max_dates}" if model.max_dates > 1 else "")
    script = f"window.{name}=" + json.dumps(blob, separators=(",", ":")) + ";"
    html = DECK.read_text(encoding="utf-8")
    pat = re.compile(rf"(<!-- {name}:BEGIN -->\n<script>).*?(</script>\n<!-- {name}:END -->)", re.S)
    if not pat.search(html):
        raise SystemExit(f"No {name}:BEGIN/END markers in {DECK}.")
    DECK.write_text(pat.sub(lambda m: m.group(1) + script + m.group(2), html), encoding="utf-8")
    dates = "one date" if model.max_dates == 1 else f"up to {model.max_dates} dates"
    print(f"Wrote the Transformer ({dates}, {count(model):,} weights, {len(script) / 1024:.0f} KB) "
          f"into {DECK} as {name}")


# =============================================================================
# COMMAND LINE
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def model_args(sp):
        sp.add_argument("--d", type=int, default=64, help="size of each position's vector")
        sp.add_argument("--layers", type=int, default=2, help="blocks in the encoder (and in the decoder)")
        sp.add_argument("--max-dates", type=int, default=1, help="read up to this many dates per input")

    t = sub.add_parser("train", help="train the encoder and decoder together")
    model_args(t)
    t.add_argument("--heads", type=int, default=4, help="attention heads per sublayer")
    t.add_argument("--ff", type=int, default=256, help="hidden size of the feed-forward sublayer")
    t.add_argument("--steps", type=int, default=4000, help="number of training rounds")
    t.add_argument("--batch", type=int, default=128, help="inputs per round")
    t.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("test", help="grade the saved model on dates and years it never saw")
    model_args(e)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_test)

    tr = sub.add_parser("translate", help="rewrite a date as YYYY-MM-DD")
    model_args(tr)
    tr.add_argument("text", help='a date, e.g. "23 Oct 2026"')
    tr.add_argument("--beam", type=int, default=1, help="beam width; 1 = greedy only")
    tr.add_argument("--details", action="store_true", help="show where cross-attention looked")
    tr.add_argument("--layer", type=int, default=0, help="which decoder block to show (default: the last)")
    tr.add_argument("--head", type=int, default=0, help="which head to show (default: average of all)")
    tr.set_defaults(func=cmd_translate)

    s = sub.add_parser("speed", help="time one training round, GRU + attention vs Transformer")
    s.add_argument("--batch", type=int, default=128)
    s.add_argument("--rounds", type=int, default=10)
    s.set_defaults(func=cmd_speed)

    x = sub.add_parser("export", help="write the trained weights into slides/transformer.html")
    model_args(x)
    x.set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
