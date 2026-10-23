# Recurrent Neural Networks, from the hidden state to LSTMs and GRUs

Course material for learning RNNs with intuition first: three slide decks, a set of interactive lessons, and a small PyTorch demo that trains RNNs, LSTMs and GRUs on Shakespeare.

```
slides/      three talks (reveal.js, one HTML file each)
demo/        char_rnn.py and word_rnn.py: train, test and play with real models
lessons/     five interactive lessons, one HTML page each
reference/   a one-page cheat sheet
```

## Slides

Each deck is one self-contained HTML file. Open it in a browser; it needs an internet connection only to load reveal.js and the fonts.

```sh
open slides/rnn-intro.html      # Part 1
open slides/lstm-gru.html       # Part 2
open slides/seq2seq.html        # Part 3
```

| Deck | Slides | Covers |
|---|---|---|
| **Part 1 · [Networks that remember](slides/rnn-intro.html)** | 24 | Why order and length need a new kind of network, the hidden state, shared weights, the three task shapes, training through time, and why blame fades. Ends with the Shakespeare demo: its results, and a slide where you type text and the model guesses the next letter. |
| **Part 2 · [Memory with a door](slides/lstm-gru.html)** | 24 | Why plain RNNs forget, where the LSTM and GRU came from, how their gates work (cell diagrams, step-by-step equations), why blame survives along the LSTM's belt, measured results from the demo, and which cell to choose. |
| **Part 3 · [Read first, then write](slides/seq2seq.html)** | 20 | Sequence to sequence: why translation needs an encoder and a decoder, the context vector, start and end tokens, training end to end, teacher forcing, greedy vs beam search, the fixed-size bottleneck, and attention as the fix. Ends with how Transformers kept the encoder–decoder shape and dropped the recurrence. |

Several slides are interactive: sliders for gates, a "you are the gates" puzzle, and a blame race between a plain RNN and an LSTM.

Keys: **→ / ←** to move, **S** for speaker notes (each slide has notes for the presenter), **Esc** for an overview of all slides, **F** for full screen.

## Demo

Two scripts that train a real model on about 1.1M characters of Shakespeare and let you test it, ask it for guesses and let it write. They run on a laptop CPU in about 25–80 seconds.

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
