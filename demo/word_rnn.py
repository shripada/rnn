# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "numpy"]
# ///
"""
==============================================================================
 A word-level RNN that learns to predict (and write) Shakespeare
==============================================================================

THE PROBLEM WE ARE SOLVING
--------------------------
    Given the words so far, predict the NEXT WORD.

    "to be or not to"          -> next is probably "be"
    "good morrow, my"          -> "lord"
    "O Romeo, Romeo! wherefore art" -> "thou"

This is the same language-modelling task as char_rnn.py, but each step of the
RNN now reads a whole WORD instead of one character. Same model, same training
loop, same four commands. Only the "alphabet" changes.

WHAT CHANGES WHEN THE UNIT IS A WORD
------------------------------------
  1. Splitting text into tokens. We lowercase everything and split it into
     words and punctuation:  "ROMEO:\\nIs she dead?"  ->
         romeo  :  ↵  is  she  dead  ?
     Each punctuation mark and each newline (↵) is its own token, so the
     model still learns the layout of a play.

  2. A much bigger vocabulary. 65 characters becomes ~4,300 words. A word
     must appear at least 3 times in TRAIN to get its own id; rarer words all
     become one shared token, <unk> ("unknown"). That covers 96% of the text.

  3. The input: an "embedding" instead of a one-hot vector. The embedding is
     a learned table E with one row of 128 numbers per word; reading a word
     means looking up its row. The RNN cell then multiplies that row by its
     own input matrix W_ih, as in Lesson 1. So Lesson 1's W_xh is split in
     two: W_xh = W_ih · Eᵀ (256 × 4,300). It can't express anything new, but
     every word is squeezed through 128 numbers, which saves weights and pushes
     words used in similar ways ("lord", "liege") toward similar rows.
     E starts random and is learned along with everything else. See them
     with the `neighbours` command.

  4. Keeping the best weights. A word model has far more weights per piece of
     text, so it starts MEMORISING TRAIN sooner: train loss keeps falling
     while val loss turns back up. `train` saves the weights from the round
     with the lowest val loss. This is what VAL is for.

  5. Shorter sequences, harder guesses. A sentence is ~10 words but ~50
     characters, so 35 steps of words reach back as far as ~175 characters.
     But each guess is now 1 of ~4,300 instead of 1 of 65, so expect far
     lower accuracy than the character model. That is normal.

THE DATA, SPLIT THREE WAYS (by token position, as in char_rnn.py)
-----------------------------------------------------------------
    first 80%  TRAIN   ~235k tokens: the model learns from these
    next  10%  VAL     check-up during training, never trained on
    last  10%  TEST    the final exam, used only by `test`

THE FOUR COMMANDS
-----------------
  uv run demo/word_rnn.py train                          # ~70 s on a laptop CPU
  uv run demo/word_rnn.py test
  uv run demo/word_rnn.py predict "to be or not to"
  uv run demo/word_rnn.py generate "romeo :" --n 80
  uv run demo/word_rnn.py neighbours lord thou king    # bonus: what the embedding learned

  Every command takes --cell rnn | lstm | gru (default lstm here: see below).
  Models are saved to demo/word-model-<cell>.pt.

WHY THE DEFAULT CELL IS LSTM
----------------------------
  A word model benefits from remembering further back (who is speaking, what
  the sentence is about), and the vanilla RNN's gradient vanishes (Lesson 3).
  Try `--cell rnn` to compare. On this small dataset the gap is small: in one
  run, words guessed right were 7.8% (rnn), 8.4% (lstm), 8.1% (gru).
"""

import argparse
import re
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
UNK = "<unk>"
MIN_COUNT = 3   # a word needs this many appearances in TRAIN to get its own id


def model_path(cell: str) -> Path:
    return HERE / f"word-model-{cell}.pt"


# =============================================================================
# 1. DATA: split text into word tokens, give each word an id, split the data
# =============================================================================

def load_text() -> str:
    if not DATA_PATH.exists():
        print(f"Downloading tinyshakespeare -> {DATA_PATH}")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def tokenize(text: str) -> list[str]:
    """Lowercase, then split into words (keeping "o'er", "thou'rt" whole),
    single punctuation marks, and newlines.

        "ROMEO:\\nIs she dead?"  ->  ['romeo', ':', '\\n', 'is', 'she', 'dead', '?']
    """
    return re.findall(r"[a-z]+(?:'[a-z]+)*|\n|[^\sa-z]", text.lower())


def split_tokens(tokens: list[str]):
    a, b = int(0.8 * len(tokens)), int(0.9 * len(tokens))
    return tokens[:a], tokens[a:b], tokens[b:]


class Vocab:
    """Every word that appears at least MIN_COUNT times in TRAIN gets an id.
    Everything else maps to the shared <unk> token (id 0)."""

    def __init__(self, words):
        self.words = list(words)                                # id -> word
        self.stoi = {w: i for i, w in enumerate(self.words)}    # word -> id

    @classmethod
    def from_tokens(cls, tokens):
        counts = Counter(tokens)
        kept = sorted(w for w, n in counts.items() if n >= MIN_COUNT)
        return cls([UNK] + kept)

    def encode(self, tokens) -> torch.Tensor:
        return torch.tensor([self.stoi.get(w, 0) for w in tokens], dtype=torch.long)

    def __len__(self):
        return len(self.words)


def detokenize(tokens) -> str:
    """Glue tokens back into readable text: no space before punctuation or
    after a newline."""
    out = ""
    for w in tokens:
        if w == "\n" or re.fullmatch(r"[^\sa-z<]", w) or not out or out[-1] in "\n-":
            out += w
        else:
            out += " " + w
    return out


def show(w: str) -> str:
    return {"\n": "↵"}.get(w, w)


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int):
    """Random chunks of `seq_len` tokens; the target is the input shifted by one word.

        x:  to   be   or   not  to
        y:  be   or   not  to   be
    """
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[s : s + seq_len] for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x, y


# =============================================================================
# 2. MODEL: the same RNN as char_rnn.py, with an embedding in front
# =============================================================================

class WordRNN(nn.Module):
    """
        e_t  = E[word at step t]                the word's row of the embedding table
        h_t  = cell(e_t, h_{t-1})               rnn / lstm / gru, as in Lessons 1, 4, 5
                                                (the cell multiplies e_t by its own W_ih)
        y_t  = W_hy · h_t   -> one score per word in the vocabulary (~4,300)
    """

    def __init__(self, vocab_size: int, embed: int, hidden: int, cell: str = "lstm"):
        super().__init__()
        self.cell = cell
        # E: one learned row of `embed` numbers per word. Row i = the vector for
        # word id i. Starts random; backprop nudges the rows of words in each batch.
        self.embed = nn.Embedding(vocab_size, embed)
        # PyTorch starts each value at random size ~1. Adam moves a weight only
        # ~lr per round, so rows that big barely move before training stops and
        # stay near-random. Starting them small lets the learned part dominate.
        nn.init.normal_(self.embed.weight, std=0.1)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(input_size=embed, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        all_h, h = self.rnn(self.embed(x), h)
        return self.head(all_h), h


def save_model(model: WordRNN, vocab: Vocab, args):
    torch.save({"state_dict": model.state_dict(), "words": vocab.words, "embed": args.embed,
                "hidden": args.hidden, "cell": model.cell}, model_path(model.cell))


def load_model(cell: str):
    path = model_path(cell)
    if not path.exists():
        raise SystemExit(f"No trained model at {path}. Run `train --cell {cell}` first.")
    ckpt = torch.load(path, weights_only=True)
    vocab = Vocab(ckpt["words"])
    model = WordRNN(len(vocab), ckpt["embed"], ckpt["hidden"], ckpt["cell"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab


# =============================================================================
# 3. A NO-MEMORY BASELINE: guess from the last word alone
# =============================================================================

class LastWordOnly:
    """"After 'my', what word usually comes next?" Counted from TRAIN.
    The best you can do with no memory of anything before the last word."""

    def __init__(self, ids: list[int]):
        self.counts = defaultdict(Counter)
        for a, b in zip(ids, ids[1:]):
            self.counts[a][b] += 1

    def top_guesses(self, last_id: int, top: int):
        c = self.counts[last_id]
        total = sum(c.values()) or 1
        return [(i, n / total) for i, n in c.most_common(top)]

    def best_guess(self, last_id: int) -> int:
        c = self.counts[last_id]
        return c.most_common(1)[0][0] if c else 0


# =============================================================================
# 4. USING THE MODEL
# =============================================================================

def encode_prompt(vocab: Vocab, prompt: str) -> torch.Tensor:
    toks = tokenize(prompt.replace("\\n", "\n"))
    if not toks:
        raise SystemExit("Give at least one word.")
    unknown = [w for w in toks if w not in vocab.stoi]
    if unknown:
        print(f"(Not in the vocabulary, read as {UNK}: {', '.join(unknown)})")
    return vocab.encode(toks)


@torch.no_grad()
def top_guesses(model: WordRNN, ids: torch.Tensor, top: int):
    logits, _ = model(ids.unsqueeze(0))
    p, i = torch.softmax(logits[0, -1], -1).topk(top)
    return list(zip(i.tolist(), p.tolist()))


@torch.no_grad()
def generate_ids(model: WordRNN, ids: torch.Tensor, n: int, temperature: float = 0.8) -> list[int]:
    """Read the prompt, then sample a word, feed it back in, repeat (one-to-many).
    <unk> is never sampled, so the output is all real words."""
    was_training = model.training
    model.eval()
    logits, h = model(ids.unsqueeze(0))
    out = ids.tolist()
    for _ in range(n):
        scores = logits[0, -1] / temperature
        scores[0] = -float("inf")                      # never write <unk>
        nxt = torch.multinomial(torch.softmax(scores, -1), 1).item()
        out.append(nxt)
        logits, h = model(torch.tensor([[nxt]]), h)
    model.train(was_training)
    return out


def fmt_guesses(vocab: Vocab, guesses):
    return ", ".join(f"{show(vocab.words[i])} {p:.0%}" for i, p in guesses)


# =============================================================================
# 5. TRAIN
# =============================================================================

def cmd_train(args):
    torch.manual_seed(args.seed)
    tokens = tokenize(load_text())
    train_toks, val_toks, test_toks = split_tokens(tokens)
    vocab = Vocab.from_tokens(train_toks)          # built from TRAIN only
    train, val = vocab.encode(train_toks), vocab.encode(val_toks)
    coverage = (train != 0).float().mean().item()

    print("THE PROBLEM: read text one WORD at a time and predict the next word.")
    x, y = get_batch(train, 1, 12)
    print("One training example (the target is the input shifted by one word):")
    print("  input  x: " + " ".join(show(vocab.words[i]) for i in x[0].tolist()))
    print("  target y: " + " ".join(show(vocab.words[i]) for i in y[0].tolist()))
    print(f"\nData: {len(train_toks):,} tokens to TRAIN on · {len(val_toks):,} VAL · {len(test_toks):,} TEST")
    print(f"Vocabulary: {len(vocab):,} words (seen ≥{MIN_COUNT}× in TRAIN), covering {coverage:.1%} "
          f"of TRAIN; the rest become {UNK}.")

    model = WordRNN(len(vocab), args.embed, args.hidden, args.cell)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {CELLS[args.cell]}, embedding {args.embed}, hidden {args.hidden}, {n_params:,} weights.")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"\nTRAINING: {args.steps} rounds of guess -> check -> adjust.")
    print(f"  random guessing among {len(vocab):,} words gives loss ln({len(vocab)}) = {torch.tensor(len(vocab)).log():.2f}")
    print("  e^loss ≈ how many words it's still choosing between (the 'perplexity')")
    print(f"{'round':>6} {'train loss':>11} {'val loss':>9} {'e^val':>7}")

    # VAL decides which weights we keep: the ones from the round with the lowest
    # val loss. Past that point the model is memorising TRAIN, not learning.
    t0, best, best_step = time.time(), float("inf"), 0
    for step in range(1, args.steps + 1):
        x, y = get_batch(train, args.batch, args.seq_len)
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, len(vocab)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)   # Lesson 3
        opt.step()

        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                vx, vy = get_batch(val, 512, args.seq_len)
                vloss = loss_fn(model(vx)[0].reshape(-1, len(vocab)), vy.reshape(-1)).item()
            mark = ""
            if vloss < best:
                best, best_step, mark = vloss, step, "  ← best so far, saved"
                save_model(model, vocab, args)
            print(f"{step:>6} {loss.item():>11.3f} {vloss:>9.3f} {torch.tensor(vloss).exp():>7.0f}{mark}")
        if step % 1000 == 0 or step == args.steps:
            prompt = vocab.encode(["romeo", ":", "\n"])
            print(f"  ── what it writes after {step} rounds (prompt 'romeo:') ──")
            text = detokenize(vocab.words[i] for i in generate_ids(model, prompt, 40))
            print("  " + text.replace("\n", "\n  "))
            print("  " + "─" * 48)

    print(f"\nTrained in {time.time() - t0:.0f}s.")
    print("Where val loss turns back up while train loss keeps falling, the model has")
    print("started memorising TRAIN instead of learning patterns that carry over.")
    print(f"Kept the weights from round {best_step} (val loss {best:.3f}) in {model_path(args.cell)}")


# =============================================================================
# 6. TEST
# =============================================================================

@torch.no_grad()
def cmd_test(args):
    model, vocab = load_model(args.cell)
    tokens = tokenize(load_text())
    train_toks, _val, test_toks = split_tokens(tokens)
    baseline = LastWordOnly(vocab.encode(train_toks).tolist())
    data = vocab.encode(test_toks)

    L = 100
    n = (len(data) - 1) // L
    x = data[: n * L].view(n, L)
    y = data[1 : n * L + 1].view(n, L)
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1)).item()
    top5 = logits.topk(5, -1).indices
    pred = top5[..., 0]
    acc = (pred == y).float().mean().item()
    acc5 = (top5 == y.unsqueeze(-1)).any(-1).float().mean().item()

    base_pred = torch.tensor([[baseline.best_guess(i) for i in row] for row in x.tolist()])
    base_acc = (base_pred == y).float().mean().item()

    # Accuracy on real words only: punctuation and newlines are easy points.
    is_word = torch.tensor([bool(re.fullmatch(r"[a-z].*", w)) for w in vocab.words])[y]
    word_acc = (pred == y)[is_word].float().mean().item()
    base_word_acc = (base_pred == y)[is_word].float().mean().item()

    print(f"TEST: grading on {n * L:,} tokens of Shakespeare the model has never seen.\n")
    print("How often is the top guess for the next token exactly right?")
    print(f"{'':32}{'all tokens':>11}{'words only':>12}")
    print(f"  {'random guessing (1 in ' + f'{len(vocab):,})':<30}{1 / len(vocab):>11.2%}{1 / len(vocab):>12.2%}")
    print(f"  {'last-word-only guesser':<30}{base_acc:>11.1%}{base_word_acc:>12.1%}   (no memory)")
    print(f"  {'your trained ' + CELLS[model.cell]:<30}{acc:>11.1%}{word_acc:>12.1%}")
    print(f"\n  right answer in the model's top 5: {acc5:.1%}")
    print(f"  loss {loss:.2f}  ->  e^loss ≈ {torch.tensor(loss).exp():.0f} words it's still choosing between "
          f"(out of {len(vocab):,})")

    print("\nSome test lines. Under each token: the model's guess, having read everything before it.")
    print("'·' = guessed right.")
    g = torch.Generator().manual_seed(args.seed)
    for _ in range(3):
        r = torch.randint(0, n, (1,), generator=g).item()
        s = torch.randint(0, L - 12, (1,), generator=g).item()
        actual = [show(vocab.words[i]) for i in y[r, s : s + 12].tolist()]
        guess = [show(vocab.words[i]) for i in pred[r, s : s + 12].tolist()]
        widths = [max(len(a), len(b)) for a, b in zip(actual, guess)]
        print("\n  actual : " + " ".join(a.ljust(w) for a, w in zip(actual, widths)))
        print("  guessed: " + " ".join(("·" if a == b else b).ljust(w) for a, b, w in zip(actual, guess, widths)))


# =============================================================================
# 7. PREDICT and GENERATE
# =============================================================================

def cmd_predict(args):
    model, vocab = load_model(args.cell)
    baseline = LastWordOnly(vocab.encode(split_tokens(tokenize(load_text()))[0]).tolist())
    ids = encode_prompt(vocab, args.text)
    print(f"Words so far: {' '.join(show(vocab.words[i]) for i in ids.tolist())}\n")
    print(f"  trained {CELLS[model.cell] + ':':<28}{fmt_guesses(vocab, top_guesses(model, ids, args.top))}")
    print(f"  last-word-only guesser (no memory): {fmt_guesses(vocab, baseline.top_guesses(ids[-1].item(), args.top))}")


def cmd_generate(args):
    model, vocab = load_model(args.cell)
    torch.manual_seed(args.seed)
    ids = encode_prompt(vocab, args.text)
    print(detokenize(vocab.words[i] for i in generate_ids(model, ids, args.n, args.temperature)))


@torch.no_grad()
def cmd_neighbours(args):
    """Which words have embedding rows closest to this word's row?

    Closeness is cosine similarity: 1 = pointing the same way, 0 = unrelated.
    Nobody told the model what words mean. Words land near each other only
    because they're used in similar places, so swapping one for the other
    barely changes the next-word guesses."""
    model, vocab = load_model(args.cell)
    E = F.normalize(model.embed.weight, dim=-1)        # each row scaled to length 1
    for word in args.words:
        word = word.lower()
        if word not in vocab.stoi:
            print(f"{word}: not in the vocabulary\n")
            continue
        sims = E @ E[vocab.stoi[word]]
        sims[vocab.stoi[word]] = -1                    # skip the word itself
        s, ids = sims.topk(args.top)
        print(f"{word}: " + ", ".join(f"{show(vocab.words[i])} {v:.2f}" for i, v in zip(ids.tolist(), s.tolist())))


# =============================================================================
# COMMAND LINE
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    cell_help = "rnn (Lessons 1-3), lstm (Lesson 4) or gru (Lesson 5)"

    t = sub.add_parser("train", help="train a fresh model and save it to demo/word-model-<cell>.pt")
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--embed", type=int, default=128, help="size of each word's learned vector")
    t.add_argument("--hidden", type=int, default=256, help="hidden state size n")
    t.add_argument("--seq-len", type=int, default=35, help="words per snippet = how far BPTT reaches back")
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--clip", type=float, default=1.0, help="gradient clipping threshold (Lesson 3)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--cell", choices=CELLS, default="lstm", help=cell_help)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("test", help="grade the saved model on the unseen TEST text")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--cell", choices=CELLS, default="lstm", help=cell_help)
    e.set_defaults(func=cmd_test)

    pr = sub.add_parser("predict", help="show the model's top guesses for the next word")
    pr.add_argument("text", help='words so far, e.g. "to be or not to"')
    pr.add_argument("--top", type=int, default=5)
    pr.add_argument("--cell", choices=CELLS, default="lstm", help=cell_help)
    pr.set_defaults(func=cmd_predict)

    g = sub.add_parser("generate", help="let the model continue a piece of text")
    g.add_argument("text", nargs="?", default="romeo :\\n", help="starting words")
    g.add_argument("--n", type=int, default=80, help="tokens to write")
    g.add_argument("--temperature", type=float, default=0.8, help="<1 safer, >1 wilder")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--cell", choices=CELLS, default="lstm", help=cell_help)
    g.set_defaults(func=cmd_generate)

    nb = sub.add_parser("neighbours", help="words whose learned embeddings are closest to a word")
    nb.add_argument("words", nargs="+", help='e.g. lord thou king')
    nb.add_argument("--top", type=int, default=8)
    nb.add_argument("--cell", choices=CELLS, default="lstm", help=cell_help)
    nb.set_defaults(func=cmd_neighbours)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
