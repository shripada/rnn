# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "numpy"]
# ///
"""
==============================================================================
 The date translator from seq2seq.py, with attention
==============================================================================

THE PROBLEM
-----------
The same task as seq2seq.py (Part 3): read a date written any of six ways,
write it as YYYY-MM-DD.

    "23 Oct 2026"                                     -> "2026-10-23"
    "the twenty-third of October, twenty twenty-six"  -> "2026-10-23"

Part 3's model squeezed the whole input into ONE context vector c, and it broke
when the input got longer: one date through c was 99% right, two dates 1%.

THE FIX: ATTENTION (Bahdanau, Cho and Bengio, 2014)
---------------------------------------------------
    ENCODER  the same GRU, but we keep its memory after EVERY character,
             h_1 ... h_T, not just the last one.
    DECODER  the same GRU, started from s_0 = h_T as before. But at every step,
             before writing, it LOOKS BACK at all of h_1 ... h_T:

      1. score    e_j = v · tanh(W_s s + W_h h_j)   how useful is input j right now?
      2. weights  a_j = softmax(e)_j                 they are positive and add up to 1
      3. context  c_t = Σ_j a_j h_j                   a fresh c for THIS step
      4. write    s = GRU([y_prev ; c_t], s)          then P(next char) from [s ; c_t]

The weights a_j are the "attention": where the decoder looked to write each
character. `translate --details` prints them as a map.

THE COMMANDS
------------
  train      build and train the model, save it
  test       grade it on unseen dates and unseen years (same exam as seq2seq.py)
  translate  give it a date; --details shows where it looked for each character
  export     put the trained weights into slides/attention.html (run it for
             the default model and for --max-dates 4)

  uv run demo/attention.py train                       # ~2 min on a laptop CPU
  uv run demo/attention.py test
  uv run demo/attention.py translate "23 Oct 2026" --details
  uv run demo/attention.py export

THE BOTTLENECK, AGAIN
---------------------
  uv run demo/attention.py train --max-dates 4         # 1-4 dates per input
  uv run demo/attention.py test  --max-dates 4
  uv run demo/attention.py translate "23 Oct 2026; May 5, 1990" --max-dates 4 --details

  Compare with `seq2seq.py train/test --max-dates 4`: same data, same exam.

HOW THIS MAPS TO THE SLIDES (Part 4)
------------------------------------
  keep every h_j        AttnSeq2Seq.encode returns all of them, not just c
  score, softmax, sum   AttnSeq2Seq.attend
  a fresh c each step   decode_step feeds c_t into the decoder and the head
  the attention map     `translate --details`
"""

import argparse
import base64
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The data, alphabets and greedy / beam search are exactly the ones in seq2seq.py.
from seq2seq import (END, FORMATS, IN_CHARS, LENGTH_BUCKETS, OUT_CHARS, PAD, all_right,
                     beam_search, encode_in, example, grade, make_batch, max_out, show, split_dates,
                     target, test_many, write_date)

HERE = Path(__file__).parent
DECK = HERE.parent / "slides" / "attention.html"


def model_path(hidden: int, max_dates: int = 1) -> Path:
    """demo/attention-gru-128.pt, demo/attention-gru-128-x4.pt, ..."""
    return HERE / (f"attention-gru-{hidden}" + (f"-x{max_dates}" if max_dates > 1 else "") + ".pt")


# =============================================================================
# 1. MODEL: seq2seq.py's encoder and decoder, plus a look back at every h_j
# =============================================================================

class AttnSeq2Seq(nn.Module):
    """
        ENCODER   h_j = GRU(x_j, h_{j-1})                 keep h_1 ... h_T
        DECODER   s_0 = h_T                               (the old c: a head start)
                  e_j = v · tanh(W_s s_{t-1} + W_h h_j)   score every input position
                  a   = softmax(e)                        attention weights
                  c_t = Σ_j a_j h_j                       this step's context
                  s_t = GRU([y_{t-1} ; c_t], s_{t-1})
                  P(next char) = softmax(W_out [s_t ; c_t])

    The decoder's state is a tuple (s, all h_j, W_h h_j, mask, last weights),
    so greedy() and beam_search() from seq2seq.py work on it unchanged.
    """

    def __init__(self, hidden: int, max_dates: int = 1, attn: int = 64):
        super().__init__()
        self.hidden, self.max_dates, self.attn = hidden, max_dates, attn
        self.encoder = nn.GRU(len(IN_CHARS), hidden, batch_first=True)
        self.decoder = nn.GRU(len(OUT_CHARS) + hidden, hidden, batch_first=True)
        self.W_h = nn.Linear(hidden, attn, bias=False)   # "what's here?"   (one per input position)
        self.W_s = nn.Linear(hidden, attn)               # "what do I need?" (from the decoder)
        self.v = nn.Linear(attn, 1, bias=False)          # turns the match into one score
        self.head = nn.Linear(2 * hidden, len(OUT_CHARS))

    def encode(self, x, lengths):
        """Read each input to its real end. Return the decoder's starting state:
        s_0 = h_T, plus every h_j for it to look back at."""
        packed = nn.utils.rnn.pack_padded_sequence(
            F.one_hot(x, len(IN_CHARS)).float(), lengths, batch_first=True, enforce_sorted=False)
        out, h_T = self.encoder(packed)
        hs, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)          # (n, T, hidden)
        mask = torch.arange(hs.shape[1])[None, :] < lengths[:, None]             # False on padding
        return h_T, hs, self.W_h(hs), mask, None

    def attend(self, s, hs, keys, mask):
        """Score every input position against the decoder's memory s, softmax the
        scores into weights, and return the weighted sum of the h_j."""
        e = self.v(torch.tanh(keys + self.W_s(s)[:, None, :]))[..., 0]           # (n, T)
        a = e.masked_fill(~mask, float("-inf")).softmax(-1)                      # padding gets 0
        return (a[..., None] * hs).sum(1), a                                     # c_t: (n, hidden)

    def decode_step(self, y_prev, state):
        """One decoder step: look back, then feed in the previous character + c_t."""
        s, hs, keys, mask, _ = state
        c_t, a = self.attend(s[0], hs, keys, mask)
        inp = torch.cat([F.one_hot(y_prev, len(OUT_CHARS)).float(), c_t], -1)
        out, s = self.decoder(inp[:, None, :], s)
        return self.head(torch.cat([out[:, 0], c_t], -1)), (s, hs, keys, mask, a)

    def forward(self, x, lengths, y_in, teacher_forcing: bool = True):
        """Scores at every output step: (batch, steps, len(OUT_CHARS)), as in seq2seq.py."""
        state = self.encode(x, lengths)
        y_prev, scores = y_in[:, 0], []
        for t in range(y_in.shape[1]):
            logits, state = self.decode_step(y_prev, state)
            scores.append(logits)
            if t + 1 < y_in.shape[1]:
                y_prev = y_in[:, t + 1] if teacher_forcing else logits.argmax(-1)
        return torch.stack(scores, 1)


def load_model(args):
    path = model_path(args.hidden, args.max_dates)
    if not path.exists():
        raise SystemExit(f"No trained model at {path}. Run `train{flags(args)}` first.")
    ckpt = torch.load(path, weights_only=True)
    model = AttnSeq2Seq(ckpt["hidden"], ckpt["max_dates"], ckpt["attn"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def flags(args) -> str:
    return ((f" --hidden {args.hidden}" if args.hidden != 128 else "")
            + (f" --max-dates {args.max_dates}" if args.max_dates != 1 else ""))


# =============================================================================
# 2. GREEDY, keeping the attention weights so we can show them
# =============================================================================

@torch.no_grad()
def greedy_with_weights(model: AttnSeq2Seq, text: str):
    """Like seq2seq.greedy: returns the answer, the top 3 guesses at each step,
    and the attention weights at each step (one row per written character)."""
    x = torch.tensor([encode_in(text)])
    state = model.encode(x, torch.tensor([x.shape[1]]))
    y_prev, out, steps, weights = torch.tensor([0]), "", [], []
    for _ in range(max_out(model.max_dates)):
        logits, state = model.decode_step(y_prev, state)
        weights.append(state[-1][0].tolist())
        best = logits.softmax(-1)[0].topk(3)
        steps.append([(OUT_CHARS[i], v) for v, i in zip(best.values.tolist(), best.indices.tolist())])
        y_prev = best.indices[:1]
        if OUT_CHARS[y_prev.item()] == END:
            break
        out += OUT_CHARS[y_prev.item()]
    return out, steps, weights


SHADES = " ·░▒▓█"


def attention_map(text: str, steps, weights) -> str:
    """One row per character written, one column per character read; darker = more weight."""
    lines = ["          " + text, "          " + "‾" * len(text)]
    for guesses, a in zip(steps, weights):
        ch, p = guesses[0]
        row = "".join(SHADES[min(5, int(w * 6 + 0.5)) if w >= 0.05 else 0] for w in a)
        j = int(np.argmax(a))
        lines.append(f"  {show(ch):>4} ←  {row}   looked at {text[j]!r} (#{j + 1}) {a[j]:.0%}")
    return "\n".join(lines)


# =============================================================================
# 3. TRAIN: the same recipe as seq2seq.py
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

    model = AttnSeq2Seq(args.hidden, args.max_dates, args.attn)
    n = lambda *mods: sum(p.numel() for m in mods for p in m.parameters())
    print(f"\nModel: GRU encoder ({n(model.encoder):,} weights) + attention "
          f"({n(model.W_h, model.W_s, model.v):,}) + GRU decoder ({n(model.decoder, model.head):,}).")
    print(f"The decoder sees every encoder memory h_1 ... h_T ({args.hidden} numbers each), "
          "not one c.")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - 0.9 * s / args.steps)
    val = make_batch(test_days, 1000, random.Random(args.seed + 1), args.max_dates)

    print(f"\nTRAINING: {args.steps} rounds of {args.batch} random inputs, teacher forcing on.")
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
    torch.save({"state_dict": model.state_dict(), "hidden": args.hidden, "max_dates": args.max_dates,
                "attn": args.attn, "history": history}, model_path(args.hidden, args.max_dates))
    print(f"Saved the trained model to {model_path(args.hidden, args.max_dates)}")
    print(f"Next: `test{flags(args)}`, or `translate \"23 Oct 2026\"{flags(args)} --details`.")


# =============================================================================
# 4. TEST: the same exam as seq2seq.py
# =============================================================================

def cmd_test(args):
    model, _ = load_model(args)
    rng = random.Random(args.seed)
    _, test_days, test_years = split_dates()
    print(f"Model: GRU encoder-decoder with attention, h has {model.hidden} numbers.\n")
    if model.max_dates > 1:
        return test_many(model, test_days, rng, why="No single c: the decoder can look back at every date.")

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
# 5. TRANSLATE: and see where it looked
# =============================================================================

def cmd_translate(args):
    model, _ = load_model(args)
    out, steps, weights = greedy_with_weights(model, args.text)
    print(f"Input: {args.text!r}  ({len(args.text)} characters: {len(args.text)} memories to look back at)\n")
    print(f"  greedy:  {out}")
    if args.details:
        print("\n  Where it looked for each character it wrote (darker = more attention):\n")
        print(attention_map(args.text, steps, weights))
    if args.beam > 1:
        print(f"\n  beam search, width {args.beam}:")
        for i, (ans, p) in enumerate(beam_search(model, args.text, args.beam), 1):
            print(f"    {i}. {ans:<14} {p:.1%}")


# =============================================================================
# 6. EXPORT: put the weights into the Part 4 slides
# =============================================================================

def f16(t: torch.Tensor) -> str:
    return base64.b64encode(t.detach().numpy().astype("<f2").tobytes()).decode()


def cmd_export(args):
    model, _ = load_model(args)
    blob = {"in": IN_CHARS, "out": OUT_CHARS, "H": model.hidden, "A": model.attn, "maxDates": model.max_dates,
            "w": {k: {"shape": list(v.shape), "b64": f16(v)} for k, v in model.state_dict().items()}}
    # The one-date model runs the "watch where it looks" slide; the --max-dates 4 one
    # runs "four dates, no single c". Each has its own slot: ATTN_MODEL, ATTN_MODEL_X4.
    name = "ATTN_MODEL" + (f"_X{model.max_dates}" if model.max_dates > 1 else "")
    script = f"window.{name}=" + json.dumps(blob, separators=(",", ":")) + ";"
    html = DECK.read_text(encoding="utf-8")
    pat = re.compile(rf"(<!-- {name}:BEGIN -->\n<script>).*?(</script>\n<!-- {name}:END -->)", re.S)
    if not pat.search(html):
        raise SystemExit(f"No {name}:BEGIN/END markers in {DECK}.")
    DECK.write_text(pat.sub(lambda m: m.group(1) + script + m.group(2), html), encoding="utf-8")
    dates = "one date" if model.max_dates == 1 else f"up to {model.max_dates} dates"
    print(f"Wrote the attention model ({dates}, h = {model.hidden} numbers, {len(script) / 1024:.0f} KB) "
          f"into {DECK} as {name}")


# =============================================================================
# COMMAND LINE
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def model_args(sp):
        sp.add_argument("--hidden", type=int, default=128, help="size of each encoder memory h_j")
        sp.add_argument("--max-dates", type=int, default=1, help="read up to this many dates per input")

    t = sub.add_parser("train", help="train the encoder, attention and decoder together")
    model_args(t)
    t.add_argument("--attn", type=int, default=64, help="size of the scoring layer inside attention")
    t.add_argument("--steps", type=int, default=4000, help="number of training rounds")
    t.add_argument("--batch", type=int, default=128, help="inputs per round")
    t.add_argument("--lr", type=float, default=3e-3, help="learning rate")
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
    tr.add_argument("--details", action="store_true", help="show where it looked for each character")
    tr.set_defaults(func=cmd_translate)

    x = sub.add_parser("export", help="write the trained weights into slides/attention.html")
    model_args(x)
    x.set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
