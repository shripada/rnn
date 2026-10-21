# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "numpy"]
# ///
"""
==============================================================================
 A character-level RNN that learns to predict (and write) Shakespeare
==============================================================================

THE PROBLEM WE ARE SOLVING
--------------------------
    Given a piece of text so far, predict the NEXT CHARACTER.

    "ROME"            -> next is almost certainly "O"   (ROMEO)
    "KING RICHAR"     -> next is almost certainly "D"   (RICHARD)
    "hear me sp"      -> next is probably "e"           (speak)

This is called *language modelling*, done one character at a time. It is the
task behind autocomplete, and it is what modern chatbots are trained on (with
words instead of characters, and much bigger models).

WHY THIS NEEDS AN RNN
---------------------
Look at "ROME". The last character is "E". If you only knew that one character,
you'd guess "N" or "S" or "R" (the most common letters after a capital E in
Shakespeare), never "O". The right answer depends on the characters before it:
"ROM". So the model needs MEMORY of earlier characters, and it has to read text
of ANY length. A feed-forward net has neither (Lesson 1). An RNN has both: it
reads one character per step and carries a hidden state h_t, a running summary
of everything read so far.

THE DATA, SPLIT THREE WAYS
--------------------------
About 1.1 million characters of Shakespeare plays (downloaded on first run):

    first 80%  TRAIN  the model learns from this text
    next  10%  VAL    a quick check-up DURING training ("is it really learning,
                      or just memorising?"). Never trained on.
    last  10%  TEST   the final exam. Only the `test` command looks at it.
                      Never seen during training.

THE FOUR COMMANDS
-----------------
  train     build a fresh RNN, train it on TRAIN, save it to demo/model.pt
  test      load the saved model and grade it on the unseen TEST text:
            how often does it guess the next character correctly?
  predict   type your own text and see the model's top guesses for the next char
  generate  give a starting text and let the model write the rest

  uv run demo/char_rnn.py train                       # ~25 s on a laptop CPU
  uv run demo/char_rnn.py test
  uv run demo/char_rnn.py predict "KING RICHAR"
  uv run demo/char_rnn.py generate "ROMEO:" --n 300
  uv run demo/char_rnn.py train --details             # + Lesson 3 numbers

SWAPPING THE CELL (Lessons 4 and 5)
-----------------------------------
  Every command takes --cell rnn | lstm | gru (default rnn). Each cell type
  is saved to its own file, so you can train all three and compare them:

  uv run demo/char_rnn.py train --cell lstm --details # LSTM, Lesson 4
  uv run demo/char_rnn.py test  --cell lstm
  uv run demo/char_rnn.py train --cell gru --details  # GRU, Lesson 5

HOW THIS MAPS TO THE LESSONS
----------------------------
  Lesson 1  the model is h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b), with shared weights
  Lesson 2  `train` is synced many-to-many (a target at every step);
            `predict` is many-to-one (read it all, one answer);
            `generate` is one-to-many (each sampled char is fed back in)
  Lesson 3  training uses BPTT + gradient clipping; `--details` shows a probe
            measuring how much gradient reaches each step back in time
  Lesson 4  `--cell lstm`: gates + a cell state c_t that is updated by ADDING,
            so the probe's gradient survives much further back
  Lesson 5  `--cell gru`: two gates, no separate cell state, ~3/4 the weights
"""

import argparse
import math
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = HERE / "data" / "tinyshakespeare.txt"
CELLS = {"rnn": "vanilla RNN", "lstm": "LSTM", "gru": "GRU"}


def model_path(cell: str) -> Path:
    """demo/model.pt for the vanilla RNN, demo/model-lstm.pt etc. for the others."""
    return HERE / ("model.pt" if cell == "rnn" else f"model-{cell}.pt")


# =============================================================================
# 1. DATA: get the text, turn characters into numbers, split it
# =============================================================================

def load_text() -> str:
    """Download tinyshakespeare once, then read it from disk."""
    if not DATA_PATH.exists():
        print(f"Downloading tinyshakespeare -> {DATA_PATH}")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def split_text(text: str):
    """Split the text into TRAIN / VAL / TEST (80 / 10 / 10), in order.

    The split is by position, not random, so the test text is a different
    part of the plays: whole scenes the model has never read."""
    a, b = int(0.8 * len(text)), int(0.9 * len(text))
    return text[:a], text[a:b], text[b:]


class Vocab:
    """The 'alphabet' of the problem: every distinct character in the text.

    Neural networks work with numbers, so each character gets an integer id
    ('a' -> 39, 'b' -> 40, ...). tinyshakespeare has 65 distinct characters:
    letters in both cases, digits, punctuation, space and newline.
    """

    def __init__(self, chars):
        self.chars = list(chars)                                # id -> char
        self.stoi = {c: i for i, c in enumerate(self.chars)}    # char -> id

    @classmethod
    def from_text(cls, text: str):
        return cls(sorted(set(text)))

    def encode(self, s: str) -> torch.Tensor:
        unknown = sorted(set(c for c in s if c not in self.stoi))
        if unknown:
            raise SystemExit(f"These characters never appear in Shakespeare's text, so the model can't read them: {unknown}")
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    def __len__(self):
        return len(self.chars)


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int):
    """Cut `batch_size` random chunks of `seq_len` characters out of the text.

    The target y is the input x shifted ONE character to the left, so at every
    position t the job is "having read x[0..t], predict x[t+1]":

        x:  F  i  r  s  t     C  i  t
        y:  i  r  s  t     C  i  t  i

    Every step has its own target (synced many-to-many, Lesson 2), so one
    100-character chunk gives 100 training examples at once.

    Returns two integer tensors of shape (batch_size, seq_len).
    """
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[s : s + seq_len] for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x, y


# =============================================================================
# 2. MODEL: exactly the RNN from Lesson 1
# =============================================================================

class CharRNN(nn.Module):
    """
        x_t  = one-hot vector for the character at step t      (size d = 65)
        h_t  = tanh(W_xh·x_t + W_hh·h_{t-1} + b)                (size n = hidden)
        y_t  = W_hy·h_t   -> one score per possible next char   (size k = 65)

    The same W_xh, W_hh, W_hy are used at every step (parameter sharing), so
    the model can read text of any length with a fixed number of weights.
    """

    def __init__(self, vocab_size: int, hidden: int, cell: str = "rnn"):
        super().__init__()
        self.vocab_size = vocab_size
        self.cell = cell
        if cell == "rnn":
            # nn.RNN is Lesson 1's equation. PyTorch's names for the matrices:
            #   weight_ih_l0 = W_xh  (shape n × d)   "input -> hidden"
            #   weight_hh_l0 = W_hh  (shape n × n)   "hidden -> hidden", the recurrence
            # It also keeps two bias vectors (b_ih + b_hh), which act as one bias b.
            # batch_first=True means tensors are shaped (batch, time, features).
            self.rnn = nn.RNN(input_size=vocab_size, hidden_size=hidden,
                              nonlinearity="tanh", batch_first=True)
        elif cell == "lstm":
            # Lesson 4. Four sets of weights (forget, input, candidate, output),
            # stacked into one matrix: weight_ih_l0 is (4n × d), weight_hh_l0 is
            # (4n × n). So ~4× the weights of the vanilla RNN at the same n.
            # The state is now a PAIR (h_t, c_t): c_t is the cell state.
            self.rnn = nn.LSTM(input_size=vocab_size, hidden_size=hidden, batch_first=True)
        elif cell == "gru":
            # Lesson 5. Three sets of weights (reset, update, candidate): ~3×
            # the vanilla RNN. The state is just h_t, like the vanilla RNN.
            # PyTorch's GRU uses h_t = (1 − z)·h̃_t + z·h_{t-1}: z = 1 keeps the old state.
            self.rnn = nn.GRU(input_size=vocab_size, hidden_size=hidden, batch_first=True)
        else:
            raise ValueError(f"unknown cell {cell!r}")
        # W_hy (shape k × n): turns the hidden state into 65 scores ("logits"),
        # one per character. Softmax turns scores into probabilities.
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        """
        x: (batch, time) integer character ids
        h: hidden state to start from; None means h_0 = 0 (Lesson 1)

        Returns:
          logits: (batch, time, 65), a prediction for the next char at EVERY step
          h:      (1, batch, hidden), the final hidden state, so you can keep
                  reading from where you left off (used by `generate`).
                  For the LSTM it is the pair (h, c).
        """
        x_onehot = F.one_hot(x, self.vocab_size).float()   # (batch, time, 65)
        # nn.RNN runs the loop over time for us: for t = 1..T, compute h_t.
        all_h, h = self.rnn(x_onehot, h)                   # all_h: h_1..h_T
        return self.head(all_h), h


def save_model(model: CharRNN, vocab: Vocab, hidden: int):
    """Save the learned weights AND the alphabet: you need both to use the model later."""
    torch.save({"state_dict": model.state_dict(), "chars": vocab.chars, "hidden": hidden,
                "cell": model.cell}, model_path(model.cell))


def load_model(cell: str = "rnn"):
    path = model_path(cell)
    if not path.exists():
        raise SystemExit(f"No trained model at {path}. Run `train --cell {cell}` first.")
    ckpt = torch.load(path, weights_only=True)
    vocab = Vocab(ckpt["chars"])
    model = CharRNN(len(vocab), ckpt["hidden"], ckpt.get("cell", "rnn"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()   # we're only using the model now, not training it
    return model, vocab


# =============================================================================
# 3. A NO-MEMORY BASELINE to compare against
# =============================================================================

class LastCharOnly:
    """Guesses the next character from the LAST character alone, using counts
    from the training text ("after 'E', what usually comes next?").

    This is the best you can do with no memory. If the RNN beats it, the RNN
    must be using its hidden state to remember earlier characters."""

    def __init__(self, text: str):
        self.counts = defaultdict(Counter)
        for a, b in zip(text, text[1:]):
            self.counts[a][b] += 1

    def top_guesses(self, prompt: str, top: int = 3):
        c = self.counts[prompt[-1]]
        total = sum(c.values()) or 1
        return [(ch, n / total) for ch, n in c.most_common(top)]

    def best_guess(self, ch: str) -> str:
        c = self.counts[ch]
        return c.most_common(1)[0][0] if c else " "


# =============================================================================
# 4. USING THE MODEL
# =============================================================================

@torch.no_grad()
def top_guesses(model: CharRNN, vocab: Vocab, prompt: str, top: int = 3):
    """Read the whole prompt, then return the top guesses for the next char.
    Only the output at the LAST step is used: a many-to-one read-out (Lesson 2)."""
    logits, _ = model(vocab.encode(prompt).unsqueeze(0))
    probs = torch.softmax(logits[0, -1], dim=-1)
    p, ids = probs.topk(top)
    return [(vocab.chars[i], pi.item()) for i, pi in zip(ids.tolist(), p)]


@torch.no_grad()
def generate_text(model: CharRNN, vocab: Vocab, prompt: str, n: int, temperature: float = 0.8) -> str:
    """Write `n` new characters after `prompt`: the one-to-many shape (Lesson 2).

    1. Read the prompt to build up a hidden state (the model's 'memory').
    2. Turn the last output into probabilities and RANDOMLY pick a character.
       Always taking the single most likely one tends to loop ("the the the"),
       so sampling keeps it varied. temperature < 1 = safer, > 1 = wilder.
    3. Feed the chosen character back in as the next input, and repeat.
    """
    was_training = model.training
    model.eval()
    logits, h = model(vocab.encode(prompt).unsqueeze(0))       # step 1
    out = list(prompt)
    for _ in range(n):
        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1).item()                # step 2
        out.append(vocab.chars[nxt])
        logits, h = model(torch.tensor([[nxt]]), h)             # step 3: carry h forward
    model.train(was_training)
    return "".join(out)


def show(ch: str) -> str:
    """Make whitespace visible when printing a single character."""
    return {" ": "␣", "\n": "↵"}.get(ch, ch)


def fmt_guesses(guesses):
    return ", ".join(f"{show(c)} {p:.0%}" for c, p in guesses)


# =============================================================================
# 5. TRAIN
# =============================================================================

def cmd_train(args):
    torch.manual_seed(args.seed)
    text = load_text()
    train_text, val_text, _test_text = split_text(text)   # TEST is not touched here
    vocab = Vocab.from_text(text)
    train, val = vocab.encode(train_text), vocab.encode(val_text)

    print("THE PROBLEM: read text one character at a time and predict the next character.")
    x, y = get_batch(train, 1, 40)
    print("One training example (the target is the input shifted by one character):")
    print(f"  input  x: {vocab.decode(x[0].tolist())!r}")
    print(f"  target y: {vocab.decode(y[0].tolist())!r}")
    print(f"\nData: {len(train_text):,} chars to TRAIN on · {len(val_text):,} VAL · "
          f"{len(text) - len(train_text) - len(val_text):,} TEST (kept for the `test` command)")

    model = CharRNN(len(vocab), args.hidden, args.cell)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {CELLS[args.cell]}, hidden size {args.hidden}, {n_params:,} weights, all random to start.")

    if args.details:
        print_probe(gradient_probe(model, train), f"Gradient probe BEFORE training ({CELLS[args.cell]}, Lesson 3):")

    # Adam: a popular form of gradient descent. It nudges every weight in the
    # direction that lowers the loss, with an automatically tuned step size.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Cross-entropy loss = -log(probability the model gave to the CORRECT next char).
    #   Uniform guessing among 65 chars gives -log(1/65) = 4.17.
    #   e^loss ≈ "how many characters it's still choosing between":
    #   loss 1.5 -> e^1.5 ≈ 4.5 choices instead of 65.
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nTRAINING: {args.steps} rounds of guess -> check -> adjust.")
    print("  loss = how wrong the guesses are (lower is better; 4.17 = random guessing)")
    print("  val loss = the same, on VAL text it never trains on")
    header = f"{'round':>6} {'train loss':>11} {'val loss':>9}"
    print(header + (f" {'grad norm':>10} {'clipped':>8}" if args.details else ""))

    t0, clipped = time.time(), 0
    for step in range(1, args.steps + 1):
        # --- one TRAINING STEP -------------------------------------------------
        # (1) GUESS: take 64 random 100-char snippets and let the RNN read them,
        #     guessing the next character at every position.
        x, y = get_batch(train, args.batch, args.seq_len)
        logits, _ = model(x)                                   # (batch, seq_len, 65)

        # (2) CHECK: compare every guess with the real next character.
        loss = loss_fn(logits.reshape(-1, len(vocab)), y.reshape(-1))

        # (3) WORK OUT THE BLAME: backpropagation through time (Lesson 3).
        #     Autograd walks back through the unrolled chain of seq_len steps and
        #     computes, for every weight, which way to nudge it to reduce the loss.
        opt.zero_grad()
        loss.backward()

        # (4) SAFETY: gradient norm clipping. If the gradient is huge, scale it
        #     down to `--clip`, which guards against exploding gradients (Lesson 3).
        norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip if args.clip > 0 else float("inf")).item()
        clipped += args.clip > 0 and norm > args.clip

        # (5) ADJUST: nudge all the weights a little in the helpful direction.
        opt.step()
        # -----------------------------------------------------------------------

        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                vx, vy = get_batch(val, args.batch, args.seq_len)
                vloss = loss_fn(model(vx)[0].reshape(-1, len(vocab)), vy.reshape(-1)).item()
            line = f"{step:>6} {loss.item():>11.3f} {vloss:>9.3f}"
            print(line + (f" {norm:>10.2f} {clipped:>8}" if args.details else ""))
            clipped = 0
        if step % 1000 == 0 or step == args.steps:
            print(f"  ── what it writes after {step} rounds (prompt 'ROMEO:') ──")
            print("  " + generate_text(model, vocab, "ROMEO:\n", 150).replace("\n", "\n  "))
            print("  " + "─" * 48)

    print(f"\nTrained in {time.time() - t0:.0f}s.")
    if args.details:
        print_probe(gradient_probe(model, train), f"Gradient probe AFTER training ({CELLS[args.cell]}, Lesson 3):")
        print("  The faster |dL/dh| falls going back in time, the less the model can learn")
        print("  from far-back characters. That's the vanishing gradient.")

    save_model(model, vocab, args.hidden)
    print(f"\nSaved the trained model to {model_path(args.cell)}")
    flag = "" if args.cell == "rnn" else f" --cell {args.cell}"
    print(f"Next: grade it on unseen text with `test{flag}`, or try it with `predict{flag}` / `generate{flag}`.")


# =============================================================================
# 6. TEST: the final exam, on text never seen during training
# =============================================================================

@torch.no_grad()
def cmd_test(args):
    model, vocab = load_model(args.cell)
    text = load_text()
    train_text, _val_text, test_text = split_text(text)
    baseline = LastCharOnly(train_text)
    data = vocab.encode(test_text)

    # Cut the test text into back-to-back 200-char chunks and let the model read
    # each one, guessing the next character at every position.
    L = 200
    n = (len(data) - 1) // L
    x = data[: n * L].view(n, L)
    y = data[1 : n * L + 1].view(n, L)
    logits, _ = model(x)
    pred = logits.argmax(-1)                       # the model's single best guess
    loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1)).item()
    rnn_acc = (pred == y).float().mean().item()

    base_pred = torch.tensor([[vocab.stoi[baseline.best_guess(vocab.chars[c])] for c in row] for row in x.tolist()])
    base_acc = (base_pred == y).float().mean().item()

    print(f"TEST: grading on {n * L:,} characters of Shakespeare the model has never seen.\n")
    print("How often is the top guess for the next character exactly right?")
    print(f"  random guessing (1 in {len(vocab)}):    {1 / len(vocab):6.1%}")
    print(f"  last-char-only guesser:       {base_acc:6.1%}   (no memory)")
    print(f"  your trained {CELLS[model.cell] + ':':<17}{rnn_acc:6.1%}   (loss {loss:.2f})")
    print("\nThe gap between the last two is what the RNN's memory (hidden state) buys you.")

    # Show a few lines of test text with the model's guesses underneath.
    print("\nSome test lines. Under each character: what the RNN guessed it would be,")
    print("having read everything before it. '·' = guessed right.")
    g = torch.Generator().manual_seed(args.seed)
    for _ in range(4):
        r = torch.randint(0, n, (1,), generator=g).item()
        start = torch.randint(0, L - 60, (1,), generator=g).item()
        actual = vocab.decode(y[r, start : start + 60].tolist())
        guess = vocab.decode(pred[r, start : start + 60].tolist())
        marks = "".join("·" if a == b else show(b) for a, b in zip(actual, guess))
        print(f"\n  actual : {''.join(show(c) for c in actual)}")
        print(f"  guessed: {marks}")


# =============================================================================
# 7. PREDICT and GENERATE: use the model on your own text
# =============================================================================

def cmd_predict(args):
    model, vocab = load_model(args.cell)
    baseline = LastCharOnly(split_text(load_text())[0])
    prompt = args.text.replace("\\n", "\n")
    print(f"Text so far: {prompt!r}\n")
    print(f"  trained {CELLS[model.cell] + ':':<28}{fmt_guesses(top_guesses(model, vocab, prompt, args.top))}")
    print(f"  last-char-only guesser (no memory): {fmt_guesses(baseline.top_guesses(prompt, args.top))}")


def cmd_generate(args):
    model, vocab = load_model(args.cell)
    torch.manual_seed(args.seed)
    prompt = args.text.replace("\\n", "\n")
    print(generate_text(model, vocab, prompt, args.n, args.temperature))


# =============================================================================
# 8. LESSON 3 PROBE (only with `train --details`)
# =============================================================================

def gradient_probe(model: CharRNN, data: torch.Tensor, seq_len: int = 100, batch: int = 32):
    """How much learning signal from a loss at the LAST step reaches each
    earlier step? We run the RNN one step at a time so each h_t can be
    inspected, put a loss only on the final prediction, backpropagate, and
    read the size of dL/dh_t. Lesson 3 predicts it shrinks roughly like f^k."""
    x, y = get_batch(data, batch, seq_len)
    x_onehot = F.one_hot(x, model.vocab_size).float()
    state, hs = None, []
    for t in range(seq_len):
        _, state = model.rnn(x_onehot[:, t : t + 1], state)   # one step: h_{t-1} -> h_t
        h = state[0] if model.cell == "lstm" else state      # LSTM state is (h, c)
        h.retain_grad()                                       # keep dL/dh_t so we can read it
        hs.append(h)
    loss = F.cross_entropy(model.head(hs[-1][0]), y[:, -1])
    model.zero_grad()
    loss.backward()
    norms = [s.grad.norm().item() / batch for s in hs]
    model.zero_grad()                                 # don't let this affect training
    return norms


def print_probe(norms, title):
    print(f"\n{title}")
    print("  steps back   |dL/dh|    (bar: log scale, 1 '#' ≈ ×2.5)")
    T, top = len(norms), max(norms)
    for k in [0, 1, 2, 5, 10, 20, 40, 60, 99]:
        if k < T:
            v = norms[T - 1 - k]
            bar = "#" * max(0, int(25 + math.log10(max(v / top, 1e-30)) * 2.5))
            print(f"  {k:>10}   {v:9.2e}  {bar}")


# =============================================================================
# COMMAND LINE
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a fresh model and save it (demo/model.pt for --cell rnn)")
    t.add_argument("--steps", type=int, default=2000, help="number of training rounds")
    t.add_argument("--hidden", type=int, default=256, help="hidden state size n")
    t.add_argument("--seq-len", type=int, default=100,
                   help="characters per snippet = how far BPTT reaches back (truncated BPTT, Lesson 3)")
    t.add_argument("--batch", type=int, default=64, help="snippets per round")
    t.add_argument("--lr", type=float, default=3e-3, help="learning rate (size of each weight nudge)")
    t.add_argument("--clip", type=float, default=1.0, help="gradient clipping threshold, 0 = off (Lesson 3)")
    t.add_argument("--details", action="store_true", help="also show gradient norms and the Lesson 3 gradient probe")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--cell", choices=CELLS, default="rnn", help="rnn (Lessons 1-3), lstm (Lesson 4) or gru (Lesson 5)")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("test", help="grade the saved model on the unseen TEST text")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--cell", choices=CELLS, default="rnn", help="rnn (Lessons 1-3), lstm (Lesson 4) or gru (Lesson 5)")
    e.set_defaults(func=cmd_test)

    pr = sub.add_parser("predict", help="show the model's top guesses for the next character")
    pr.add_argument("text", help='text so far, e.g. "KING RICHAR" (use \\n for a newline)')
    pr.add_argument("--top", type=int, default=5)
    pr.add_argument("--cell", choices=CELLS, default="rnn", help="rnn (Lessons 1-3), lstm (Lesson 4) or gru (Lesson 5)")
    pr.set_defaults(func=cmd_predict)

    g = sub.add_parser("generate", help="let the model continue a piece of text")
    g.add_argument("text", nargs="?", default="ROMEO:\n", help="starting text")
    g.add_argument("--n", type=int, default=300, help="characters to write")
    g.add_argument("--temperature", type=float, default=0.8, help="<1 safer, >1 wilder")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--cell", choices=CELLS, default="rnn", help="rnn (Lessons 1-3), lstm (Lesson 4) or gru (Lesson 5)")
    g.set_defaults(func=cmd_generate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
