# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "numpy", "transformers>=4.40"]
# ///
"""
==============================================================================
 BERT: the Transformer's encoder on its own, pretrained by filling in blanks
==============================================================================

WHAT THIS IS
------------
Parts 3-6 trained small models from nothing. BERT is different: someone else
already trained it, for days, on 3.3 billion words, and we download it. This
script opens it up and pokes at it.

    bert-base-uncased  (Devlin and colleagues, 2018)
    12 encoder blocks, d = 768, 12 heads, feed-forward 768 -> 3072 -> 768
    a vocabulary of 30,522 WordPiece tokens, up to 512 of them per input
    about 110 million weights; downloaded once (~440 MB) by Hugging Face's
    `transformers` library and cached in ~/.cache/huggingface

THE TWO PRETRAINING TASKS
-------------------------
    MASKED LANGUAGE MODEL   hide 15% of the tokens, guess them from BOTH sides
        "the man went to the [MASK] to buy milk"  ->  store
    NEXT SENTENCE PREDICTION  was sentence B really the one after sentence A?
        A "the man went to the store."  B "he bought a gallon of milk."  -> IsNext

Then FINE-TUNING: put one small layer on top, train the whole thing a little
on your task.

THE COMMANDS
------------
  tokens     how BERT cuts text into WordPiece tokens
  fill       guess the [MASK]ed word(s); --left-only adds GPT-2's guess from
             only the words before the blank
  senses     the same word in different sentences: how alike are its vectors,
             layer by layer? (contextual embeddings)
  attention  one head's attention map for a sentence
  nsp        next sentence prediction: does B follow A?
  finetune   teach it positive / negative movie reviews (SST-2), and compare
             with a model that has no pretraining
  classify   run the fine-tuned model on your own sentence
  export     write the examples the slides use into slides/bert.html

  uv run demo/bert.py tokens "Transformers revolutionised embeddings"
  uv run demo/bert.py fill "the man went to the [MASK] to buy a gallon of milk ."
  uv run demo/bert.py fill "the [MASK] barked at the mailman all morning ." --left-only
  uv run demo/bert.py senses bank
  uv run demo/bert.py attention "the animal didn't cross the street because it was too tired ." --layer 6 --head 11
  uv run demo/bert.py nsp "the man went to the store ." "he bought a gallon of milk ."
  uv run demo/bert.py finetune                      # ~3 min on a laptop CPU
  uv run demo/bert.py classify "a gripping, beautifully acted film"
  uv run demo/bert.py export

HOW THIS MAPS TO THE SLIDES (Part 7)
------------------------------------
  WordPiece, [CLS], [SEP], segments   cmd_tokens, encode()
  masked language model               cmd_fill  (the `cls.predictions` head)
  why both directions                 cmd_fill --left-only
  next sentence prediction            cmd_nsp   (the `cls.seq_relationship` head)
  contextual embeddings               cmd_senses (hidden states of every block)
  what the heads look at              cmd_attention
  fine-tuning                         cmd_finetune, Classifier
"""

import argparse
import io
import logging
import os
import json
import random
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Quiet the download chatter, progress bars and a harmless torch notice on macOS
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

HERE = Path(__file__).parent
DECK = HERE.parent / "slides" / "bert.html"
DATA = HERE / "data"
NAME = "bert-base-uncased"
SST2_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"


def load_bert():
    """BERT with both pretraining heads still attached: the masked-word guesser
    (cls.predictions) and the next-sentence guesser (cls.seq_relationship)."""
    from transformers import BertForPreTraining, BertTokenizer
    tok = BertTokenizer.from_pretrained(NAME)
    model = BertForPreTraining.from_pretrained(NAME, attn_implementation="eager")   # eager: so it can return its attention
    model.eval()
    return tok, model


def encode(tok, a: str, b: str | None = None) -> dict:
    """[CLS] a [SEP] (b [SEP]), with segment ids 0 for a and 1 for b. The
    tokenizer understands a literal "[MASK]" in the text."""
    return tok(a, b, return_tensors="pt")


def show_tokens(tok, ids) -> list[str]:
    return tok.convert_ids_to_tokens(ids.tolist() if torch.is_tensor(ids) else ids)


# =============================================================================
# 1. TOKENS: WordPiece
# =============================================================================

def cmd_tokens(args):
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(NAME)
    enc = encode(tok, args.text, args.second)
    toks, segs = show_tokens(tok, enc["input_ids"][0]), enc["token_type_ids"][0].tolist()
    print(f"Text: {args.text!r}" + (f"  +  {args.second!r}" if args.second else ""))
    print(f"\n{len(toks)} tokens (vocabulary: {tok.vocab_size:,} pieces; '##' = joins the piece before):\n")
    print(f"  {'#':>3}  {'token':<16} {'id':>6}  segment")
    for i, (t, idx, s) in enumerate(zip(toks, enc["input_ids"][0].tolist(), segs)):
        print(f"  {i:>3}  {t:<16} {idx:>6}  {'A' if s == 0 else 'B'}")
    print("\n[CLS] starts every input (its top vector is the one classifiers use); [SEP] ends each segment.")
    for w in args.text.split():
        pieces = tok.tokenize(w)
        if len(pieces) > 1:
            print(f"  {w!r} isn't in the vocabulary as a whole: {' + '.join(pieces)}")


# =============================================================================
# 2. FILL: the masked language model
# =============================================================================

@torch.no_grad()
def guesses(tok, model, text: str, top: int = 5):
    """For each [MASK] in text: the top guesses and their probabilities."""
    enc = encode(tok, text)
    logits = model(**enc).prediction_logits[0]                  # (tokens, 30522)
    where = (enc["input_ids"][0] == tok.mask_token_id).nonzero().flatten().tolist()
    out = []
    for i in where:
        p = logits[i].softmax(-1).topk(top)
        out.append([(tok.convert_ids_to_tokens(j), v) for v, j in zip(p.values.tolist(), p.indices.tolist())])
    return out


def load_gpt():
    """GPT-2 (Radford and colleagues, OpenAI, 2019), the smallest one: also 12 blocks
    of d = 768, but decoder-only. Each position sees only the ones before it, and
    it's trained to guess the NEXT word. Here only as the contrast to BERT."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tok, model


@torch.no_grad()
def next_word(tok, model, before: str, top: int = 5):
    """GPT-2's guesses for the word after `before`. GPT-2 sees capitals (BERT-uncased
    doesn't), so the first letter is capitalised, after its start-of-text token. Its
    pieces mark a leading space with 'Ġ'; we keep only guesses that start a new word."""
    text = tok.bos_token + before[:1].upper() + before[1:]
    p = model(**tok(text, return_tensors="pt")).logits[0, -1].softmax(-1)
    best = []
    for v, j in zip(*p.topk(100)):
        t = tok.convert_ids_to_tokens(j.item())
        if not before and t[:1].isascii() and t[:1].isalpha():   # the very first word has no space before it
            t = "Ġ" + t.lower()
        if t.startswith("Ġ") and t[1:2].isalnum() and len(best) < top:
            best.append((t[1:], v.item()))
    return best


def cmd_fill(args):
    text = args.text.replace("___", "[MASK]")
    if "[MASK]" not in text:
        raise SystemExit('Put [MASK] (or ___) where the hidden word goes, e.g. "the [MASK] barked".')
    tok, model = load_bert()
    print(f"BERT, reading both sides:  {text!r}")
    for k, g in enumerate(guesses(tok, model, text, args.top), 1):
        print(f"  [MASK] {k}:  " + ", ".join(f"{w} {p:.0%}" for w, p in g))
    if args.left_only:
        before = text[: text.index("[MASK]")].rstrip()
        gtok, gpt = load_gpt()
        print(f"\nGPT-2, reading left to right, has only:  {before!r}")
        print("  next word:  " + ", ".join(f"{w} {p:.0%}" for w, p in next_word(gtok, gpt, before, args.top)))


# =============================================================================
# 3. SENSES: contextual embeddings
# =============================================================================

BANK = [("money", "i deposited the cheque at the bank this morning ."),
        ("money", "the bank approved her loan on friday ."),
        ("money", "he works as a cashier at a small bank ."),
        ("river", "we had a picnic on the grassy bank of the river ."),
        ("river", "the boat drifted slowly towards the muddy bank ."),
        ("river", "fish were hiding near the bank of the stream .")]


@torch.no_grad()
def word_vectors(tok, model, sentences, word: str):
    """The vector for `word` in each sentence, after every block: (sentences, 13, 768).
    Layer 0 is the input to block 1 (token + position + segment embedding)."""
    out = []
    for s in sentences:
        enc = encode(tok, s)
        toks = show_tokens(tok, enc["input_ids"][0])
        if word not in toks:
            raise SystemExit(f"{word!r} isn't a single token in {s!r}: tokens are {toks}")
        hs = model.bert(**enc, output_hidden_states=True).hidden_states   # 13 × (1, T, 768)
        out.append(torch.stack([h[0, toks.index(word)] for h in hs]))
    return torch.stack(out)


def cosines(v: torch.Tensor) -> torch.Tensor:
    """v: (sentences, layers, d) -> (layers, sentences, sentences) cosine similarities."""
    v = F.normalize(v.transpose(0, 1), dim=-1)
    return v @ v.transpose(1, 2)


def sense_gap(sim: torch.Tensor, labels) -> list[tuple[float, float]]:
    """Per layer: average similarity between pairs with the same meaning, and with different meanings."""
    n, same, diff = len(labels), [], []
    for layer in sim:
        s = [layer[i, j].item() for i in range(n) for j in range(i + 1, n) if labels[i] == labels[j]]
        d = [layer[i, j].item() for i in range(n) for j in range(i + 1, n) if labels[i] != labels[j]]
        same.append(np.mean(s))
        diff.append(np.mean(d))
    return list(zip(same, diff))


def cmd_senses(args):
    tok, model = load_bert()
    if args.sentences:
        labels, sents = [str(i + 1) for i in range(len(args.sentences))], args.sentences
    elif args.word == "bank":
        labels, sents = zip(*BANK)
    else:
        raise SystemExit('Give sentences with --sentences "..." "..." (built-in examples are for "bank").')
    sim = cosines(word_vectors(tok, model, sents, args.word))
    print(f"The word {args.word!r} in {len(sents)} sentences:\n")
    for i, (lab, s) in enumerate(zip(labels, sents), 1):
        print(f"  {i}. [{lab}] {s}")
    for layer in (0, args.layer):
        print(f"\nCosine similarity of the {args.word!r} vectors, "
              + ("layer 0 (embedding table + position, before any block):" if layer == 0 else f"after block {layer}:"))
        print("      " + "".join(f"{j:>6}" for j in range(1, len(sents) + 1)))
        for i in range(len(sents)):
            print(f"  {i + 1:>3} " + "".join(f"{sim[layer, i, j]:>6.2f}" for j in range(len(sents))))
    if 1 < len(set(labels)) < len(labels):                    # only when meanings are labelled
        print("\nAverage similarity, same meaning vs different meaning:")
        print(f"  {'layer':>5} {'same':>6} {'different':>10}")
        for layer, (s, d) in enumerate(sense_gap(sim, labels)):
            print(f"  {layer:>5} {s:>6.2f} {d:>10.2f}")
    print("\nA table like word2vec's gives the word ONE vector, whatever the sentence.")
    print("BERT starts from such a table (layer 0) and every block mixes in the neighbours.")


# =============================================================================
# 4. ATTENTION: what one head looks at
# =============================================================================

SHADES = " ·░▒▓█"


@torch.no_grad()
def attentions(tok, model, text: str, b: str | None = None):
    """Tokens and every head's attention: (layers, heads, T, T)."""
    enc = encode(tok, text, b)
    att = model.bert(**enc, output_attentions=True).attentions
    return show_tokens(tok, enc["input_ids"][0]), torch.stack([a[0] for a in att])


def cmd_attention(args):
    tok, model = load_bert()
    toks, att = attentions(tok, model, args.text)
    L, H = att.shape[:2]
    a = att[args.layer - 1, args.head - 1]
    w = max(len(t) for t in toks)
    print(f"Block {args.layer} of {L}, head {args.head} of {H}. Row = the token looking, column = looked at.\n")
    for i, t in enumerate(toks):
        row = "".join(SHADES[min(5, int(v * 6))] * 2 for v in a[i].tolist())
        j = a[i].argmax().item()
        print(f"  {t:>{w}}  {row}  → {toks[j]} {a[i, j]:.0%}")
    print("\n  columns: " + " ".join(f"{k}={t}" for k, t in enumerate(toks)))
    print("\n  Where every head looks hardest, averaged over the tokens, for each block:")
    for layer in range(L):
        share = att[layer].mean((0, 1))
        top = share.topk(2)
        print(f"    block {layer + 1:>2}: " + ", ".join(f"{toks[j]} {v:.0%}" for v, j in zip(top.values.tolist(), top.indices.tolist())))


# =============================================================================
# 5. NSP: next sentence prediction
# =============================================================================

@torch.no_grad()
def is_next(tok, model, a: str, b: str) -> float:
    """P(B really follows A), from the [CLS] vector through the NSP head."""
    return model(**encode(tok, a, b)).seq_relationship_logits[0].softmax(-1)[0].item()


def cmd_nsp(args):
    tok, model = load_bert()
    p = is_next(tok, model, args.a, args.b)
    print(f"A: {args.a!r}\nB: {args.b!r}\n")
    print(f"  IsNext {p:.1%}   NotNext {1 - p:.1%}")
    print(f"\n  Reversed, B then A:  IsNext {is_next(tok, model, args.b, args.a):.1%}")


# =============================================================================
# 6. FINE-TUNE: positive or negative movie review (SST-2)
# =============================================================================

def sst2():
    """The Stanford Sentiment Treebank, as packaged for the GLUE benchmark: short
    movie-review phrases labelled 1 (positive) or 0 (negative). Its test labels are
    hidden, so, as everyone does, we grade on the 872-sentence dev set."""
    zpath = DATA / "SST-2.zip"
    if not zpath.exists():
        DATA.mkdir(exist_ok=True)
        print(f"Downloading SST-2 (7 MB) to {zpath} ...")
        urllib.request.urlretrieve(SST2_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        def read(name):
            lines = io.TextIOWrapper(z.open(f"SST-2/{name}.tsv"), encoding="utf-8").read().splitlines()[1:]
            return [(s, int(y)) for s, y in (ln.rsplit("\t", 1) for ln in lines)]
        return read("train"), read("dev")


class Classifier(nn.Module):
    """BERT plus ONE new layer: the top block's [CLS] vector -> 2 scores.

        h = BERT(text)[CLS]          768 numbers, from the pretrained encoder
        P(positive) = softmax(h W + b)   W: 768 × 2, the only weights that start random

    Fine-tuning trains W, b AND all of BERT's weights, gently (small learning rate).
    With --frozen, BERT's weights stay as downloaded and only W, b learn."""

    def __init__(self, bert):
        super().__init__()
        self.bert, self.drop, self.out = bert, nn.Dropout(0.1), nn.Linear(bert.config.hidden_size, 2)

    def forward(self, **enc):
        h = self.bert(**enc).last_hidden_state[:, 0]            # the [CLS] position
        return self.out(self.drop(h))


class BagOfWords(nn.Module):
    """No pretraining: each word gets its own learned vector, from random, and a
    sentence is the average of its words' vectors -> 2 scores. Word order is ignored."""

    def __init__(self, vocab: int, d: int = 64):
        super().__init__()
        self.emb, self.out = nn.EmbeddingBag(vocab, d, mode="mean"), nn.Linear(d, 2)

    def forward(self, ids, offsets):
        return self.out(self.emb(ids, offsets))


def finetuned_dir(frozen: bool) -> Path:
    return HERE / ("bert-sst2-frozen" if frozen else "bert-sst2")


def batches(data, n: int, rng: random.Random):
    data = data[:]
    rng.shuffle(data)
    return [data[i: i + n] for i in range(0, len(data), n)]


@torch.no_grad()
def accuracy_bert(model, tok, data, n: int = 64) -> float:
    model.eval()
    right = 0
    for i in range(0, len(data), n):
        chunk = data[i: i + n]
        enc = tok([s for s, _ in chunk], padding=True, truncation=True, max_length=64, return_tensors="pt")
        right += (model(**enc).argmax(-1) == torch.tensor([y for _, y in chunk])).sum().item()
    return right / len(data)


def train_bag(train, dev, args):
    """The no-pretraining baseline, trained on the same examples."""
    torch.manual_seed(args.seed)
    words = lambda s: re.findall(r"[a-z']+|[^\sa-z']", s.lower())
    vocab = {w: i + 1 for i, w in enumerate(sorted({w for s, _ in train for w in words(s)}))}   # 0 = unknown
    def tensors(chunk):
        ids = [torch.tensor([vocab.get(w, 0) for w in words(s)] or [0]) for s, _ in chunk]
        offsets = torch.tensor([0] + [len(t) for t in ids[:-1]]).cumsum(0)
        return torch.cat(ids), offsets, torch.tensor([y for _, y in chunk])
    model = BagOfWords(len(vocab) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = random.Random(args.seed)
    for _ in range(30):                                          # it's tiny: 30 passes take seconds
        for chunk in batches(train, 32, rng):
            ids, off, y = tensors(chunk)
            loss = F.cross_entropy(model(ids, off), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        ids, off, y = tensors(dev)
        return (model(ids, off).argmax(-1) == y).float().mean().item(), len(vocab)


def cmd_finetune(args):
    from transformers import BertModel, BertTokenizer
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train_all, dev = sst2()
    train = rng.sample(train_all, args.examples)
    print(f"SST-2: movie-review phrases, positive (1) or negative (0). {len(train_all):,} in the training set; "
          f"we use {len(train):,} of them, and grade on the {len(dev)} dev sentences.")
    for s, y in train[:4]:
        print(f"  {'positive' if y else 'negative':<9} {s}")

    print("\n1. NO PRETRAINING: average of learned word vectors, from random, same examples.")
    t0 = time.time()
    bag_acc, nv = train_bag(train, dev, args)
    print(f"   {nv:,} words in its vocabulary. Dev accuracy {bag_acc:.1%}  ({time.time() - t0:.0f}s)")

    tok = BertTokenizer.from_pretrained(NAME)
    model = Classifier(BertModel.from_pretrained(NAME, add_pooling_layer=False))
    if args.frozen:
        for p in model.bert.parameters():
            p.requires_grad = False
    learn = [p for p in model.parameters() if p.requires_grad]
    print(f"\n2. BERT + one new layer ({model.out.weight.numel() + model.out.bias.numel():,} new weights), "
          + ("BERT FROZEN: only the new layer learns." if args.frozen else
             f"FINE-TUNED: all {sum(p.numel() for p in learn):,} weights learn."))
    print(f"   Before training (random top layer): dev accuracy {accuracy_bert(model, tok, dev):.1%}")

    # The paper's recipe: Adam, learning rate 2e-5 to 5e-5, 2 to 4 passes, warm-up then linear decay.
    lr = args.lr or (1e-3 if args.frozen else 3e-5)
    opt = torch.optim.AdamW(learn, lr=lr, weight_decay=0.01)
    steps = args.epochs * -(-len(train) // args.batch)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else (steps - s) / (steps - warm))
    print(f"   {args.epochs} passes over {len(train):,} examples, batch {args.batch}, learning rate {lr:g}: {steps} rounds.\n")
    print(f"  {'round':>6} {'train loss':>11} {'dev accuracy':>13} {'time':>6}")
    t0, step, history = time.time(), 0, []
    for _ in range(args.epochs):
        for chunk in batches(train, args.batch, rng):
            model.train()
            enc = tok([s for s, _ in chunk], padding=True, truncation=True, max_length=64, return_tensors="pt")
            loss = F.cross_entropy(model(**enc), torch.tensor([y for _, y in chunk]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(learn, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % args.every == 0 or step == steps:
                acc = accuracy_bert(model, tok, dev)
                history.append((step, round(loss.item(), 4), round(acc, 4)))
                print(f"  {step:>6} {loss.item():>11.3f} {acc:>13.1%} {time.time() - t0:>5.0f}s")

    out = finetuned_dir(args.frozen)
    out.mkdir(exist_ok=True)
    if not args.frozen:                                          # frozen BERT is the download: keep only the new layer
        model.bert.save_pretrained(out)
    torch.save({"head": model.out.state_dict(), "bag_acc": bag_acc, "history": history, "examples": len(train),
                "frozen": args.frozen}, out / "head.pt")
    print(f"\nNo pretraining: {bag_acc:.1%}.  BERT {'frozen' if args.frozen else 'fine-tuned'}: {history[-1][2]:.1%}.")
    print(f"Saved to {out}/. Next: `classify \"a gripping, beautifully acted film\"`.")


def load_classifier(frozen: bool = False):
    from transformers import BertModel, BertTokenizer
    path = finetuned_dir(frozen)
    if not (path / "head.pt").exists():
        raise SystemExit(f"No fine-tuned model at {path}. Run `finetune{' --frozen' if frozen else ''}` first.")
    model = Classifier(BertModel.from_pretrained(NAME if frozen else path, add_pooling_layer=False))
    model.out.load_state_dict(torch.load(path / "head.pt", weights_only=True)["head"])
    model.eval()
    return BertTokenizer.from_pretrained(NAME), model


@torch.no_grad()
def positive(tok, model, text: str) -> float:
    return model(**tok(text, return_tensors="pt")).softmax(-1)[0, 1].item()


def cmd_classify(args):
    tok, model = load_classifier(args.frozen)
    for t in args.text:
        p = positive(tok, model, t)
        print(f"  {'positive' if p >= 0.5 else 'negative':<9} {max(p, 1 - p):6.1%}   {t}")


# =============================================================================
# 7. EXPORT: the examples the slides use
# =============================================================================

FILL = ["the man went to the store to buy a gallon of milk .",
        "the dog barked at the mailman all morning .",
        "she put the cake in the oven and waited for it to bake .",
        "paris is the capital and largest city of france .",
        "he scored the winning goal in the final minute of the match ."]
ATTENTION = ["the animal didn't cross the street because it was too tired .",
             "the animal didn't cross the street because it was too wide ."]
NSP_A = ["the man went to the store .", "she opened the old wooden door .", "the match ended in a draw ."]
NSP_B = ["he bought a gallon of milk .", "behind it was a dusty staircase .", "neither team managed to score .",
         "penguins cannot fly ."]
REVIEWS = ["a gripping, beautifully acted film .", "a dull, plodding mess .",
           "not bad at all .", "not good at all .", "i expected to hate it , but i loved it .",
           "the plot is thin , but the performances are wonderful ."]


def pct(t) -> list:
    """Probabilities as whole-number tenths of a percent, to keep the slides small."""
    return [int(round(v * 1000)) for v in (t.tolist() if torch.is_tensor(t) else t)]


@torch.no_grad()
def cmd_export(args):
    tok, model = load_bert()
    blob = {"vocab": [tok.convert_ids_to_tokens(i) for i in range(tok.vocab_size)]}

    # every single-piece word of each FILL sentence masked in turn: BERT with both
    # sides, GPT-2 with only the words before
    gtok, gpt = load_gpt()
    fill = []
    for s in FILL:
        toks = s.split()
        rows = []
        for i, w in enumerate(toks):
            if len(tok.tokenize(w)) != 1:
                rows.append(None)
                continue
            both = guesses(tok, model, " ".join(toks[:i] + ["[MASK]"] + toks[i + 1:]))[0]
            left = next_word(gtok, gpt, " ".join(toks[:i]))
            rows.append({"bert": [[t, round(p, 3)] for t, p in both], "gpt": [[t, round(p, 3)] for t, p in left]})
        fill.append({"words": toks, "rows": rows})
    blob["fill"] = fill

    # "bank": similarity at every layer
    labels, sents = zip(*BANK)
    sim = cosines(word_vectors(tok, model, sents, "bank"))
    blob["senses"] = {"labels": labels, "sentences": sents, "sim": [pct(layer.flatten()) for layer in sim],
                      "gap": [[round(s, 3), round(d, 3)] for s, d in sense_gap(sim, labels)]}

    # every head's attention for two sentences that differ in one word
    blob["attention"] = []
    for s in ATTENTION:
        toks, att = attentions(tok, model, s)
        blob["attention"].append({"tokens": toks, "a": [[pct(att[l, h].flatten()) for h in range(att.shape[1])]
                                                       for l in range(att.shape[0])]})

    # next sentence prediction for every A, B pair
    blob["nsp"] = {"a": NSP_A, "b": NSP_B, "p": [[round(is_next(tok, model, a, b), 3) for b in NSP_B] for a in NSP_A]}

    # the fine-tuned classifier, if there is one
    path = finetuned_dir(False) / "head.pt"
    if path.exists():
        ctok, cls = load_classifier()
        info = torch.load(path, weights_only=True)
        blob["sst2"] = {"reviews": [[r, round(positive(ctok, cls, r), 3)] for r in REVIEWS],
                        "history": info["history"], "bag": round(info["bag_acc"], 4), "examples": info["examples"]}
        frozen = finetuned_dir(True) / "head.pt"
        if frozen.exists():
            blob["sst2"]["frozen"] = torch.load(frozen, weights_only=True)["history"][-1][2]
    else:
        print(f"(No fine-tuned model at {path.parent}: run `finetune` first to include the classifier.)")

    script = "window.BERT_DATA=" + json.dumps(blob, separators=(",", ":"), ensure_ascii=False) + ";"
    html = DECK.read_text(encoding="utf-8")
    pat = re.compile(r"(<!-- BERT_DATA:BEGIN -->\n<script>).*?(</script>\n<!-- BERT_DATA:END -->)", re.S)
    if not pat.search(html):
        raise SystemExit(f"No BERT_DATA:BEGIN/END markers in {DECK}.")
    DECK.write_text(pat.sub(lambda m: m.group(1) + script + m.group(2), html), encoding="utf-8")
    print(f"Wrote BERT's vocabulary and the slide examples ({len(script) / 1024:.0f} KB) into {DECK}")


# =============================================================================
# COMMAND LINE
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("tokens", help="show how BERT cuts text into WordPiece tokens")
    t.add_argument("text")
    t.add_argument("second", nargs="?", help="an optional second sentence (segment B)")
    t.set_defaults(func=cmd_tokens)

    f = sub.add_parser("fill", help="guess the [MASK]ed words")
    f.add_argument("text", help='text with [MASK] (or ___) for each hidden word')
    f.add_argument("--left-only", action="store_true", help="also guess from only the words before the first blank")
    f.add_argument("--top", type=int, default=5)
    f.set_defaults(func=cmd_fill)

    s = sub.add_parser("senses", help="one word in many sentences: how alike are its vectors?")
    s.add_argument("word", nargs="?", default="bank")
    s.add_argument("--sentences", nargs="+", help="your own sentences, each containing the word")
    s.add_argument("--layer", type=int, default=12, help="which block's output to print in full (1-12)")
    s.set_defaults(func=cmd_senses)

    a = sub.add_parser("attention", help="one head's attention map")
    a.add_argument("text")
    a.add_argument("--layer", type=int, default=1, help="block, 1-12")
    a.add_argument("--head", type=int, default=1, help="head, 1-12")
    a.set_defaults(func=cmd_attention)

    n = sub.add_parser("nsp", help="next sentence prediction: does B follow A?")
    n.add_argument("a")
    n.add_argument("b")
    n.set_defaults(func=cmd_nsp)

    ft = sub.add_parser("finetune", help="fine-tune BERT on SST-2 movie reviews")
    ft.add_argument("--examples", type=int, default=2000, help="training sentences to use (of 67,349)")
    ft.add_argument("--epochs", type=int, default=2)
    ft.add_argument("--batch", type=int, default=16)
    ft.add_argument("--lr", type=float, default=0, help="default 3e-5 (1e-3 with --frozen)")
    ft.add_argument("--frozen", action="store_true", help="keep BERT's weights fixed; train only the new layer")
    ft.add_argument("--every", type=int, default=25, help="check dev accuracy every N rounds")
    ft.add_argument("--seed", type=int, default=0)
    ft.set_defaults(func=cmd_finetune)

    c = sub.add_parser("classify", help="positive or negative? (needs `finetune` first)")
    c.add_argument("text", nargs="+")
    c.add_argument("--frozen", action="store_true", help="use the --frozen model")
    c.set_defaults(func=cmd_classify)

    x = sub.add_parser("export", help="write the slide examples into slides/bert.html")
    x.set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
