# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "numpy"]
# ///
"""
==============================================================================
 A sequence-to-sequence RNN that rewrites dates as YYYY-MM-DD
==============================================================================

THE PROBLEM WE ARE SOLVING
--------------------------
    Read a date written the way people write it, and write it the way a
    computer wants it:

    "23 Oct 2026"                                     -> "2026-10-23"
    "Friday, 23 October 2026"                         -> "2026-10-23"
    "the twenty-third of October, twenty twenty-six"  -> "2026-10-23"

This is a tiny TRANSLATION task. The input and the output have different
lengths, different alphabets and a different order (the day comes first in,
last out), so the synced many-to-many RNN of the Shakespeare demo can't do it.
It needs Part 3's shape: an ENCODER that reads the whole input, and a DECODER
that writes the output from what the encoder left behind.

THE MODEL
---------
    ENCODER  a GRU reads the input one character at a time. Its final hidden
             state is the CONTEXT VECTOR c: everything it knows about the date.
    DECODER  a second GRU, with its own weights, starts from h_0 = c. It is
             given the start token ⟨s⟩, writes one character, is fed that
             character back in, and so on until it writes the end token ⟨/s⟩.

The only link between the two is c: a fixed list of numbers (128 by default),
however long the input is. That's the bottleneck from the slides.

THE DATA
--------
Generated in code: random dates in six formats, some in lower or upper case.

    TRAIN  dates from 1900-2079 (except every 10th day)
    TEST   (a) every 10th day of 1900-2079: dates it never saw
           (b) every day of 2080-2099: YEARS it never saw
           Only the `test` command looks at these.

"23/10/2026" is never in training at all. Try it with `translate`.

THE COMMANDS
------------
  train      build a fresh encoder + decoder, train them together, save them
  test       grade the saved model on the unseen dates: by format, by input
             length, and on the unseen years
  translate  give it a date and watch it write the answer
  export     put the trained weights into slides/seq2seq.html for the live slides

  uv run demo/seq2seq.py train                         # ~70 s on a laptop CPU
  uv run demo/seq2seq.py test
  uv run demo/seq2seq.py translate "23 Oct 2026" --details
  uv run demo/seq2seq.py translate "SEP 13 2088" --beam 3
  uv run demo/seq2seq.py export

EXPERIMENTS FROM THE SLIDES
---------------------------
  uv run demo/seq2seq.py train --max-dates 4           # the bottleneck: 1-4 dates
  uv run demo/seq2seq.py test  --max-dates 4           #   through the same c (~5 min)
  uv run demo/seq2seq.py train --no-teacher-forcing    # feed it its own guesses
  uv run demo/seq2seq.py train --hidden 32             # a smaller c

  Every command takes --cell rnn | lstm | gru (default gru), --hidden N
  (default 128) and --max-dates N (default 1). Each combination is saved to
  its own file, e.g. demo/seq2seq-gru-128.pt, demo/seq2seq-gru-128-x4.pt.

HOW THIS MAPS TO THE SLIDES (Part 3)
------------------------------------
  encoder              many-to-one: read everything, keep only the last h
  context vector c     the encoder's last h, the decoder's first h
  ⟨s⟩ and ⟨/s⟩          ordinary entries in the decoder's alphabet
  trained end to end   one loss at the decoder, blame flows back into the encoder
  teacher forcing      during training the decoder is fed the RIGHT previous char
  greedy vs beam       `translate --beam 3` shows both
  the bottleneck       `--max-dates 4`: more dates, same c
"""

import argparse
import base64
import datetime as dt
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
DECK = HERE.parent / "slides" / "seq2seq.html"
CELLS = {"rnn": "vanilla RNN", "lstm": "LSTM", "gru": "GRU"}

FIRST_YEAR, LAST_TRAIN_YEAR, LAST_YEAR = 1900, 2079, 2099


def model_path(cell: str, hidden: int, max_dates: int = 1, teacher_forcing: bool = True) -> Path:
    """demo/seq2seq-gru-128.pt, demo/seq2seq-gru-16.pt, demo/seq2seq-gru-128-x4.pt,
    demo/seq2seq-gru-128-notf.pt, ..."""
    return HERE / (f"seq2seq-{cell}-{hidden}" + (f"-x{max_dates}" if max_dates > 1 else "")
                   + ("" if teacher_forcing else "-notf") + ".pt")


# =============================================================================
# 1. DATA: dates written six ways
# =============================================================================

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth",
            "eight": "eighth", "nine": "ninth", "twelve": "twelfth", "twenty": "twentieth",
            "thirty": "thirtieth"}


def words(n: int) -> str:
    """1..99 in words: 7 -> 'seven', 42 -> 'forty-two'."""
    if n < 20:
        return ONES[n]
    return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")


def ordinal(n: int) -> str:
    """1..31 as an ordinal: 1 -> 'first', 23 -> 'twenty-third'."""
    w = words(n).split("-")
    last = w[-1]
    w[-1] = ORDINALS.get(last, last + "th")
    return "-".join(w)


def year_words(y: int) -> str:
    """How people say a year: 1984 'nineteen eighty-four', 1905 'nineteen oh five',
    1900 'nineteen hundred', 2006 'two thousand and six', 2026 'twenty twenty-six'."""
    hi, lo = divmod(y, 100)
    if y == 2000:
        return "two thousand"
    if 2000 < y < 2010:
        return "two thousand and " + words(lo)
    if lo == 0:
        return words(hi) + " hundred"
    if lo < 10:
        return words(hi) + " oh " + words(lo)
    return words(hi) + " " + words(lo)


FORMATS = {
    "23 Oct 2026":             lambda d: f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}",
    "23 October 2026":         lambda d: f"{d.day} {MONTHS[d.month - 1]} {d.year}",
    "October 23, 2026":        lambda d: f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
    "Oct 23 2026":             lambda d: f"{MONTHS[d.month - 1][:3]} {d.day} {d.year}",
    "Friday, 23 October 2026": lambda d: f"{DAYS[d.weekday()]}, {d.day} {MONTHS[d.month - 1]} {d.year}",
    "the twenty-third of …":   lambda d: f"the {ordinal(d.day)} of {MONTHS[d.month - 1]}, {year_words(d.year)}",
}


def write_date(d: dt.date, fmt: str, rng: random.Random) -> str:
    """One date in one format, as-is 70% of the time, else all lower or all upper case."""
    s = FORMATS[fmt](d)
    r = rng.random()
    return s.lower() if r < 0.15 else s.upper() if r < 0.30 else s


def target(d: dt.date) -> str:
    return d.isoformat()                         # "2026-10-23"


def all_days(first: int, last: int):
    d, end = dt.date(first, 1, 1), dt.date(last, 12, 31)
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def split_dates():
    """TRAIN: 1900-2079 minus every 10th day. TEST: those held-out days, plus all of 2080-2099."""
    train, test_days = [], []
    for d in all_days(FIRST_YEAR, LAST_TRAIN_YEAR):
        (test_days if d.toordinal() % 10 == 0 else train).append(d)
    test_years = list(all_days(LAST_TRAIN_YEAR + 1, LAST_YEAR))
    return train, test_days, test_years


def example(dates, k: int, rng: random.Random):
    """k random dates, each in a random format, joined with '; ' (k = 1 for the normal model):
    ("23 Oct 2026; the fifth of May, nineteen ninety", "2026-10-23;1990-05-05")"""
    ds = [rng.choice(dates) for _ in range(k)]
    return ("; ".join(write_date(d, rng.choice(list(FORMATS)), rng) for d in ds),
            ";".join(target(d) for d in ds))


# The encoder's alphabet: everything the six formats can produce, ';' between dates,
# plus '/' and '.' so you can try "23/10/2026" (never seen in training: those columns
# of W_xh stay random).
IN_CHARS = " ,-./0123456789;ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# The decoder's alphabet: ⟨s⟩ (start), ⟨/s⟩ (end), and what YYYY-MM-DD dates are made of.
START, END = "^", "$"
OUT_CHARS = START + END + "-0123456789;"
PAD = -100                                       # "no target here": the loss skips it


def max_out(max_dates: int) -> int:
    """Stop writing after this many characters, ⟨/s⟩ or not."""
    return 11 * max_dates + 3


def show(ch: str) -> str:
    return {START: "⟨s⟩", END: "⟨/s⟩"}.get(ch, ch)


def encode_in(s: str) -> list[int]:
    unknown = sorted(set(c for c in s if c not in IN_CHARS))
    if unknown:
        raise SystemExit(f"The encoder can't read these characters: {unknown}. It knows: {IN_CHARS!r}")
    return [IN_CHARS.index(c) for c in s]


def make_batch(dates, n: int, rng: random.Random, max_dates: int = 1):
    """n random examples of 1..max_dates dates each, turned into tensors."""
    return pairs_to_tensors([example(dates, rng.randint(1, max_dates), rng) for _ in range(n)])


def pairs_to_tensors(pairs):
    """
    x       (n, longest input)  input character ids, padded with 0 at the end
    lengths (n,)                how long each input really is
    y_in    (n, 11)             what the decoder is FED:    ⟨s⟩ 2 0 2 6 - 1 0 - 2 3
    y_out   (n, 11)             what it should WRITE:       2 0 2 6 - 1 0 - 2 3 ⟨/s⟩
    y_in is y_out shifted right by one, exactly like x and y in the Shakespeare demo.
    (With several dates per input the outputs differ in length: the short ones are
    padded, y_out with PAD so the loss ignores those steps.)"""
    lengths = torch.tensor([len(s) for s, _ in pairs])
    x = torch.zeros(len(pairs), int(lengths.max()), dtype=torch.long)
    T = max(len(t) for _, t in pairs) + 1
    y_in = torch.full((len(pairs), T), OUT_CHARS.index(END), dtype=torch.long)
    y_out = torch.full((len(pairs), T), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(pairs):
        x[i, : len(s)] = torch.tensor(encode_in(s))
        y_in[i, : len(t) + 1] = torch.tensor([OUT_CHARS.index(c) for c in START + t])
        y_out[i, : len(t) + 1] = torch.tensor([OUT_CHARS.index(c) for c in t + END])
    return x, lengths, y_in, y_out


def all_right(scores, y_out):
    """True where every character of the answer is right (padding doesn't count)."""
    return ((scores.argmax(-1) == y_out) | (y_out == PAD)).all(1)


# =============================================================================
# 2. MODEL: two RNNs joined by c
# =============================================================================

def make_cell(cell: str, inp: int, hidden: int) -> nn.Module:
    # The same three cells as the Shakespeare demo (Parts 1 and 2).
    if cell == "rnn":
        return nn.RNN(inp, hidden, nonlinearity="tanh", batch_first=True)
    if cell == "lstm":
        return nn.LSTM(inp, hidden, batch_first=True)
    if cell == "gru":
        return nn.GRU(inp, hidden, batch_first=True)
    raise ValueError(f"unknown cell {cell!r}")


class Seq2Seq(nn.Module):
    """
        ENCODER   h_t = f(x_t, h_{t-1}),  h_0 = 0         reads the date
                  c   = h_T                               the context vector
        DECODER   s_0 = c
                  s_t = g(y_{t-1}, s_{t-1})               y_0 = ⟨s⟩
                  P(next char) = softmax(W_out · s_t)

    f and g are separate RNNs with separate weights. Inputs are one-hot,
    like the Shakespeare demo. For the LSTM, c is the pair (h, cell state).
    """

    def __init__(self, hidden: int, cell: str = "gru", max_dates: int = 1):
        super().__init__()
        self.cell, self.hidden, self.max_dates = cell, hidden, max_dates
        self.encoder = make_cell(cell, len(IN_CHARS), hidden)
        self.decoder = make_cell(cell, len(OUT_CHARS), hidden)
        self.head = nn.Linear(hidden, len(OUT_CHARS))

    def encode(self, x, lengths):
        """Read each input to its real end (not into the padding) and return c."""
        packed = nn.utils.rnn.pack_padded_sequence(
            F.one_hot(x, len(IN_CHARS)).float(), lengths, batch_first=True, enforce_sorted=False)
        _, c = self.encoder(packed)
        return c

    def decode_step(self, y_prev, state):
        """One decoder step: feed in the previous character, get scores for the next."""
        out, state = self.decoder(F.one_hot(y_prev, len(OUT_CHARS)).float().unsqueeze(1), state)
        return self.head(out[:, 0]), state

    def forward(self, x, lengths, y_in, teacher_forcing: bool = True):
        """Returns the decoder's scores at every output step: (batch, steps, len(OUT_CHARS)).

        teacher_forcing=True:  at each step feed the RIGHT previous character (y_in).
        teacher_forcing=False: feed the model's own best guess from the step before,
                               exactly as it will have to when nobody knows the answer."""
        state = self.encode(x, lengths)
        y_prev, scores = y_in[:, 0], []
        for t in range(y_in.shape[1]):
            logits, state = self.decode_step(y_prev, state)
            scores.append(logits)
            if t + 1 < y_in.shape[1]:
                y_prev = y_in[:, t + 1] if teacher_forcing else logits.argmax(-1)
        return torch.stack(scores, 1)


def save_model(model: Seq2Seq, extra: dict):
    torch.save({"state_dict": model.state_dict(), "hidden": model.hidden, "cell": model.cell,
                "max_dates": model.max_dates, **extra},
               model_path(model.cell, model.hidden, model.max_dates, extra["teacher_forcing"]))


def load_model(args):
    path = model_path(args.cell, args.hidden, args.max_dates)
    if not path.exists():
        raise SystemExit(f"No trained model at {path}. Run `train{flags(args)}` first.")
    ckpt = torch.load(path, weights_only=True)
    model = Seq2Seq(ckpt["hidden"], ckpt["cell"], ckpt.get("max_dates", 1))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# =============================================================================
# 3. WRITING THE ANSWER: greedy and beam search
# =============================================================================

@torch.no_grad()
def greedy(model: Seq2Seq, text: str, top: int = 3):
    """Take the single most likely character at each step and feed it back in.
    Returns the answer and, for each step, the top guesses with their chances."""
    x = torch.tensor([encode_in(text)])
    state = model.encode(x, torch.tensor([x.shape[1]]))
    y_prev, out, steps = torch.tensor([0]), "", []
    for _ in range(max_out(model.max_dates)):
        logits, state = model.decode_step(y_prev, state)
        p = logits.softmax(-1)[0]
        best = p.topk(top)
        steps.append([(OUT_CHARS[i], v) for v, i in zip(best.values.tolist(), best.indices.tolist())])
        y_prev = best.indices[:1]
        ch = OUT_CHARS[y_prev.item()]
        if ch == END:
            break
        out += ch
    return out, steps


@torch.no_grad()
def beam_search(model: Seq2Seq, text: str, width: int = 3):
    """Keep the `width` most likely partial answers at every step, not just one.

    Each beam is (log-probability so far, characters so far, decoder state).
    At each step every beam proposes its `width` best next characters, and we
    keep the `width` best of all those. A beam that writes ⟨/s⟩ is finished.
    Returns finished answers, most likely first, as (answer, probability)."""
    x = torch.tensor([encode_in(text)])
    c = model.encode(x, torch.tensor([x.shape[1]]))
    beams, done = [(0.0, "", c)], []
    for _ in range(max_out(model.max_dates)):
        candidates = []
        for logp, out, state in beams:
            prev = OUT_CHARS.index(out[-1]) if out else 0
            logits, new_state = model.decode_step(torch.tensor([prev]), state)
            lp = logits.log_softmax(-1)[0]
            v, i = lp.topk(width)
            for vv, ii in zip(v.tolist(), i.tolist()):
                candidates.append((logp + vv, out + OUT_CHARS[ii], new_state))
        candidates.sort(key=lambda b: -b[0])
        beams = []
        for logp, out, state in candidates:
            if out.endswith(END):
                done.append((out[:-1], logp))
            elif len(beams) < width:
                beams.append((logp, out, state))
        # Stop once the best unfinished beam can't beat the width-th finished one
        # (adding characters only ever lowers a probability).
        done.sort(key=lambda b: -b[1])
        if not beams or (len(done) >= width and beams[0][0] < done[width - 1][1]):
            break
    done.sort(key=lambda b: -b[1])
    return [(out, float(np.exp(lp))) for out, lp in done[:width]]


# =============================================================================
# 4. TRAIN
# =============================================================================

def cmd_train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train_dates, test_days, _ = split_dates()

    if args.max_dates == 1:
        print("THE PROBLEM: read a date written any of six ways, write it as YYYY-MM-DD.")
        for fmt in FORMATS:
            d = rng.choice(train_dates)
            print(f"  {write_date(d, fmt, rng):<52} -> {target(d)}")
    else:
        print(f"THE PROBLEM: read 1 to {args.max_dates} dates, write them all as YYYY-MM-DD.")
        for k in range(1, args.max_dates + 1):
            s, t = example(train_dates, k, rng)
            print(f"  {s}\n    -> {t}")
    print(f"\nData: {len(train_dates):,} dates to TRAIN on (1900-2079) · "
          f"{len(test_days):,} held-out days + 2080-2099 kept for `test`")

    model = Seq2Seq(args.hidden, args.cell, args.max_dates)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters()) + sum(p.numel() for p in model.head.parameters())
    print(f"Model: {CELLS[args.cell]} encoder ({n_enc:,} weights) + {CELLS[args.cell]} decoder "
          f"({n_dec:,} weights). c has {args.hidden} numbers.")
    print(f"Teacher forcing: {'ON (fed the right previous character)' if args.teacher_forcing else 'OFF (fed its own guesses)'}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Big nudges early, small ones at the end: the learning rate falls in a straight
    # line to 1/10 of its start. Without this, some runs keep bouncing around near
    # the answer instead of settling on it.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - 0.9 * s / args.steps)
    # Validation: a fixed set of held-out days, so the numbers are comparable between rounds.
    val = make_batch(test_days, 1000, random.Random(args.seed + 1), args.max_dates)

    print(f"\nTRAINING: {args.steps} rounds. Each round: {args.batch} random inputs, the encoder")
    print("reads each one, the decoder writes the answer, ONE loss over all the output")
    print("characters, and blame flows back through c into the encoder.")
    print(f"{'round':>6} {'train loss':>11} {'val loss':>9} {'val exact':>10}")

    t0, history = time.time(), []
    for step in range(1, args.steps + 1):
        x, lengths, y_in, y_out = make_batch(train_dates, args.batch, rng, args.max_dates)
        scores = model(x, lengths, y_in, teacher_forcing=args.teacher_forcing)
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
                # Graded the way it will be used: fed its OWN guesses, no answer key.
                vs = model(vx, vl, vyi, teacher_forcing=False)
                vloss = F.cross_entropy(vs.reshape(-1, len(OUT_CHARS)), vyo.reshape(-1), ignore_index=PAD).item()
                exact = all_right(vs, vyo).float().mean().item()
            model.train()
            history.append((step, round(time.time() - t0, 1), round(loss.item(), 4), round(vloss, 4), round(exact, 4)))
            print(f"{step:>6} {loss.item():>11.3f} {vloss:>9.3f} {exact:>10.1%}")

    print(f"\nTrained in {time.time() - t0:.0f}s.")
    print("(val exact = whole answer right, writing on its own with no answer key)")
    save_model(model, {"teacher_forcing": args.teacher_forcing, "history": history})
    print(f"Saved the trained model to {model_path(args.cell, args.hidden, args.max_dates, args.teacher_forcing)}")
    print(f"Next: `test{flags(args)}`, or `translate \"23 Oct 2026\"{flags(args)} --details`.")


def flags(args) -> str:
    return ((f" --cell {args.cell}" if args.cell != "gru" else "")
            + (f" --hidden {args.hidden}" if args.hidden != 128 else "")
            + (f" --max-dates {args.max_dates}" if args.max_dates != 1 else ""))


# =============================================================================
# 5. TEST: dates and years it never saw
# =============================================================================

LENGTH_BUCKETS = [(0, 14), (15, 24), (25, 34), (35, 44), (45, 54), (55, 99)]


@torch.no_grad()
def grade(model, pairs, chunk: int = 2000):
    """Greedy-decode every input (in chunks); return a list of True/False (whole answer right)."""
    right = []
    for i in range(0, len(pairs), chunk):
        x, lengths, y_in, y_out = pairs_to_tensors(pairs[i : i + chunk])
        right += all_right(model(x, lengths, y_in, teacher_forcing=False), y_out).tolist()
    return right


def cmd_test(args):
    model, ckpt = load_model(args)
    rng = random.Random(args.seed)
    _, test_days, test_years = split_dates()
    print(f"Model: {CELLS[model.cell]}, c has {model.hidden} numbers, teacher forcing "
          f"{'on' if ckpt.get('teacher_forcing', True) else 'off'} in training.\n")
    if model.max_dates > 1:
        return test_many(model, test_days, rng)

    # Every held-out day in every format: 6 × 6,574 inputs.
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
        r = [ok for ok, ff in zip(right, fmts) if ff == f]
        print(f"  {f:<26} {np.mean(r):6.1%}")

    print("\nBy input length:")
    for lo, hi in LENGTH_BUCKETS:
        r = [ok for ok, (s, _) in zip(right, pairs) if lo <= len(s) <= hi]
        if r:
            print(f"  {lo:>2}-{hi:<2} chars  {np.mean(r):6.1%}   ({len(r):,} inputs)")

    year_pairs = [(write_date(d, f, rng), target(d)) for d in test_years for f in FORMATS]
    yr = grade(model, year_pairs)
    print(f"\nYears it never saw (2080-2099), all formats: {np.mean(yr):.1%}")
    spelled = [ok for ok, (s, _) in zip(yr, year_pairs) if s.lower().startswith("the ")]
    print(f"  of which spelled out ('twenty eighty-one' etc.): {np.mean(spelled):.1%}")

    print("\nSome mistakes (input -> wrong answer, right answer):")
    wrong = [(s, t) for ok, (s, t) in zip(right + yr, pairs + year_pairs) if not ok]
    for s, t in rng.sample(wrong, min(6, len(wrong))):
        print(f"  {s:<52} -> {greedy(model, s)[0]:<12} should be {t}")
    if not wrong:
        print("  none")


def test_many(model, test_days, rng, per_k: int = 3000,
              why: str = "Every date has to fit through the same c."):
    """The bottleneck test: the same c has to carry 1, 2, 3, ... dates.
    (attention.py runs the same test on its model, with its own `why`.)"""
    print(f"TEST: {per_k:,} inputs for each number of dates, all built from dates it never saw.")
    print(f"{why} How often is the WHOLE answer right,")
    print("how often is each date in it right, and which dates does it get right?\n")
    print(f"  {'dates':>5} {'avg input':>10} {'whole answer':>13} {'each date':>10}   by position (1st, 2nd, ...)")
    for k in range(1, model.max_dates + 1):
        pairs = [example(test_days, k, rng) for _ in range(per_k)]
        right, by_pos = grade(model, pairs), [[] for _ in range(k)]
        for i in range(0, len(pairs), 2000):
            x, lengths, y_in, y_out = pairs_to_tensors(pairs[i : i + 2000])
            with torch.no_grad():
                pred = model(x, lengths, y_in, teacher_forcing=False).argmax(-1)
            for j in range(k):   # date j sits at output characters 11j .. 11j+9
                by_pos[j] += (pred[:, 11 * j : 11 * j + 10] == y_out[:, 11 * j : 11 * j + 10]).all(1).tolist()
        avg_len = np.mean([len(s) for s, _ in pairs])
        each = np.mean([ok for pos in by_pos for ok in pos])
        print(f"  {k:>5} {avg_len:>8.0f} ch {np.mean(right):>12.1%} {each:>10.1%}   "
              + "  ".join(f"{np.mean(pos):.0%}" for pos in by_pos))


# =============================================================================
# 6. TRANSLATE: try your own date
# =============================================================================

def cmd_translate(args):
    model, _ = load_model(args)
    print(f"Input: {args.text!r}  ({len(args.text)} characters, squeezed into c: {model.hidden} numbers)\n")
    out, steps = greedy(model, args.text)
    print(f"  greedy:  {out}")
    if args.details:
        print("\n  What the decoder thought at each step (top 3):")
        for t, guesses in enumerate(steps, 1):
            print(f"    step {t:>2}: " + ", ".join(f"{show(c)} {p:.0%}" for c, p in guesses))
    if args.beam > 1:
        print(f"\n  beam search, width {args.beam}:")
        for i, (ans, p) in enumerate(beam_search(model, args.text, args.beam), 1):
            print(f"    {i}. {ans:<14} {p:.1%}")


# =============================================================================
# 7. EXPORT: put the weights into the Part 3 slides
# =============================================================================

def f16(t: torch.Tensor) -> str:
    """A weight matrix as base64 float16: half the size of float32, same answers."""
    return base64.b64encode(t.detach().numpy().astype("<f2").tobytes()).decode()


def cmd_export(args):
    model, _ = load_model(args)
    if model.cell != "gru" or model.max_dates != 1:
        raise SystemExit("The slides run a one-date GRU. Export a model trained with --cell gru --max-dates 1.")
    blob = {"in": IN_CHARS, "out": OUT_CHARS, "H": model.hidden,
            "w": {k: {"shape": list(v.shape), "b64": f16(v)} for k, v in model.state_dict().items()}}
    script = "window.SEQ2SEQ_MODEL=" + json.dumps(blob, separators=(",", ":")) + ";"
    html = DECK.read_text(encoding="utf-8")
    pat = re.compile(r"(<!-- SEQ2SEQ_MODEL:BEGIN -->\n<script>).*?(</script>\n<!-- SEQ2SEQ_MODEL:END -->)", re.S)
    if not pat.search(html):
        raise SystemExit(f"No SEQ2SEQ_MODEL:BEGIN/END markers in {DECK}.")
    DECK.write_text(pat.sub(lambda m: m.group(1) + script + m.group(2), html), encoding="utf-8")
    print(f"Wrote {CELLS[model.cell]} weights (c = {model.hidden} numbers, {len(script) / 1024:.0f} KB) into {DECK}")


# =============================================================================
# COMMAND LINE
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def model_args(sp):
        sp.add_argument("--cell", choices=CELLS, default="gru", help="rnn, lstm or gru (default)")
        sp.add_argument("--hidden", type=int, default=128, help="hidden size n = the size of c")
        sp.add_argument("--max-dates", type=int, default=1,
                        help="read up to this many dates per input (the bottleneck experiment)")

    t = sub.add_parser("train", help="train a fresh encoder + decoder and save them")
    model_args(t)
    t.add_argument("--steps", type=int, default=4000, help="number of training rounds")
    t.add_argument("--batch", type=int, default=128, help="inputs per round")
    t.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    t.add_argument("--no-teacher-forcing", dest="teacher_forcing", action="store_false",
                   help="feed the decoder its own guesses during training, not the right answers")
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
    tr.add_argument("--details", action="store_true", help="show the top 3 guesses at every decoder step")
    tr.set_defaults(func=cmd_translate)

    x = sub.add_parser("export", help="write the trained GRU's weights into slides/seq2seq.html")
    model_args(x)
    x.set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
