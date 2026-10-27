# Recurrent Neural Networks, from the hidden state to LSTMs and GRUs

Course material for learning RNNs with intuition first: five slide decks, a set of interactive lessons, and small PyTorch demos that train RNNs, LSTMs and GRUs on Shakespeare, and GRU encoder–decoders, with and without attention, that translate dates.

```
slides/      four talks (reveal.js, one HTML file each)
demo/        char_rnn.py, word_rnn.py, seq2seq.py and attention.py: train, test and play with real models
lessons/     five interactive lessons, one HTML page each
reference/   a one-page cheat sheet
```

## Slides

Each deck is one self-contained HTML file. Open it in a browser; it needs an internet connection only to load reveal.js and the fonts.

```sh
open slides/rnn-intro.html      # Part 1
open slides/lstm-gru.html       # Part 2
open slides/seq2seq.html        # Part 3
open slides/attention.html      # Part 4
open slides/self-attention.html # Part 5
```

| Deck | Slides | Covers |
|---|---|---|
| **Part 1 · [Networks that remember](slides/rnn-intro.html)** | 24 | Why order and length need a new kind of network, the hidden state, shared weights, the three task shapes, training through time, and why blame fades. Ends with the Shakespeare demo: its results, and a slide where you type text and the model guesses the next letter. |
| **Part 2 · [Memory with a door](slides/lstm-gru.html)** | 24 | Why plain RNNs forget, where the LSTM and GRU came from, how their gates work (cell diagrams, step-by-step equations), why blame survives along the LSTM's belt, measured results from the demo, and which cell to choose. |
| **Part 3 · [Read first, then write](slides/seq2seq.html)** | 23 | Sequence to sequence: why translation needs an encoder and a decoder, the context vector, start and end tokens, training end to end, teacher forcing, greedy vs beam search, and the fixed-size bottleneck. Then the date-translation demo, live: step through the encoder, c and the decoder; greedy vs beam on inputs where they disagree; and the bottleneck, measured. Ends with attention as the fix. |
| **Part 4 · [Look back while you write](slides/attention.html)** | 20 | Attention: keep every encoder memory, then at each decoder step score them, soften the scores with softmax and mix them into a fresh context. A softmax playground, a step-through of “the black cat”, why attention gives blame a short path back, and alignment maps. Then the date translator with attention, live: watch where it looks for each character, and four dates at once with no bottleneck. Ends with what attention doesn't fix and why the Transformer dropped the RNN. |
| **Part 5 · [Every word looks at every word](slides/self-attention.html)** | 21 | Self-attention: drop the RNN and let the sentence attend to itself. Queries, keys and values; the scaled dot product, with a √d<sub>k</sub> playground; one row worked by hand; what “it” refers to; the whole sentence as three matrix multiplies. Then the catch, live: shuffle the words and the outputs just shuffle, fixed by adding positions. Multi-head attention with a heads explorer, masked attention for writing, one Transformer block, and the RNN vs self-attention trade. |

Several slides are interactive: sliders for gates, a "you are the gates" puzzle, a blame race between a plain RNN and an LSTM, in Parts 3 and 4 the trained date translators running in the browser, including a live attention map, and in Part 5 a √d<sub>k</sub> playground, a word-order test on a real self-attention layer and a heads explorer.

Keys: **→ / ←** to move, **S** for speaker notes (each slide has notes for the presenter), **Esc** for an overview of all slides, **F** for full screen.

## Demo

Two scripts train a real model on about 1.1M characters of Shakespeare and let you test it, ask it for guesses and let it write. A third trains a sequence-to-sequence model that rewrites dates, and a fourth adds attention to it. Each trains on a laptop CPU in about 25 seconds to 2 minutes (the multi-date experiments take longer).

You need [uv](https://docs.astral.sh/uv/). It installs Python, PyTorch and NumPy on the first run.

### `char_rnn.py`: predict the next character

```sh
uv run demo/char_rnn.py train                      # train and save the model (~25 s)
uv run demo/char_rnn.py test                       # grade it on unseen text
uv run demo/char_rnn.py predict "KING RICHAR"      # top guesses for the next character
uv run demo/char_rnn.py generate "ROMEO:" --n 300  # let it write
```

Add `--cell lstm` or `--cell gru` to any command to swap the plain RNN for a gated cell, and `--details` to `train` to see how much of the gradient survives 10, 40 and 99 steps back. Each cell saves to its own file, so you can train all three and compare them.

From one run (hidden size 256, 2000 rounds):

| | plain RNN | LSTM | GRU |
|---|---|---|---|
| next character right on unseen text | 47.1% | 50.0% | 50.3% |
| training time | 24 s | 80 s | 64 s |

A guesser that only looks at the last character gets 26.5%, so the memory is doing real work.

### `word_rnn.py`: predict the next word

The same task with whole words as the unit, an LSTM by default, and a learned embedding for each word.

```sh
uv run demo/word_rnn.py train                          # ~70 s
uv run demo/word_rnn.py predict "good morrow, my"      # lord 44%, gracious 5%, ...
uv run demo/word_rnn.py neighbours lord king she       # words the model learned to treat alike
```

### `seq2seq.py`: translate dates (Part 3)

An encoder–decoder: a GRU reads a date written any of six ways, its last memory becomes the context c, and a second GRU writes it as `YYYY-MM-DD`.

```sh
uv run demo/seq2seq.py train                           # ~70 s
uv run demo/seq2seq.py translate "23 Oct 2026" --details   # 2026-10-23, with the decoder's top guesses at each step
uv run demo/seq2seq.py translate "SEP 13 2088" --beam 3    # greedy says 2038, beam search finds 2088
uv run demo/seq2seq.py export                          # put the weights into the Part 3 slides
```

From one run: 100% right on 6,574 days it never saw (in all six formats), but only 12% on years it never saw (2080–2099). It learned the patterns it saw, not the rule. `--max-dates 4` measures the bottleneck: one date through c is 99% right, two or more are about 0%.

### `attention.py`: the date translator, with attention (Part 4)

The same task, data and exam as `seq2seq.py`. The decoder now looks back at every encoder memory at each step, instead of relying on one `c`.

```sh
uv run demo/attention.py train                              # ~2 min
uv run demo/attention.py test
uv run demo/attention.py translate "23 Oct 2026" --details  # a map of where it looked for each character
uv run demo/attention.py export                             # put the weights into the Part 4 slides
```

From one run: 100% on unseen days, as before, but also **99.8% on unseen years** (Part 3: 12%), because it can look at the `9` in `2091` and copy it. With `--max-dates 4` it gets 1, 2, 3 and 4 dates all 100% right, where Part 3's single `c` managed 99%, 1%, 0%, 0%.

[demo/README.md](demo/README.md) has the full walkthrough: the train / validation / test split, what one training round does, sample output from every command, and which lesson each part of the code matches.

Trained models (`demo/*.pt`) aren't in the repo; `train` creates them.

## Lessons

Five interactive pages, about 20 minutes each, meant to be read in order:

1. [The hidden state](lessons/0001-the-hidden-state.html)
2. [Sequence task shapes](lessons/0002-task-shapes.html)
3. [Vanishing gradients](lessons/0003-vanishing-gradients.html)
4. [LSTM: gated memory](lessons/0004-lstm.html)
5. [GRU: the lighter gate](lessons/0005-gru.html)

[reference/rnn-core.html](reference/rnn-core.html) is a one-page summary of the equations and shapes.

## Credits

The character demo follows Andrej Karpathy's [char-rnn](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) and uses its tinyshakespeare dataset. The LSTM and GRU notation follows [Dive into Deep Learning](https://d2l.ai/) and Christopher Olah's [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
