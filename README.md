# Recurrent Neural Networks, from the hidden state to LSTMs and GRUs

Course material for learning RNNs with intuition first: seven slide decks, a set of interactive lessons, and small PyTorch demos that train RNNs, LSTMs and GRUs on Shakespeare, and date translators built as GRU encoder–decoders, with and without attention, and as a Transformer, and that open up a pretrained BERT.

```
slides/      seven talks (reveal.js, one HTML file each)
demo/        char_rnn.py, word_rnn.py, seq2seq.py, attention.py, transformer.py and bert.py: train, test and play with real models
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
open slides/transformer.html    # Part 6
open slides/bert.html           # Part 7
```

| Deck | Slides | Covers |
|---|---|---|
| **Part 1 · [Networks that remember](slides/rnn-intro.html)** | 24 | Why order and length need a new kind of network, the hidden state, shared weights, the three task shapes, training through time, and why blame fades. Ends with the Shakespeare demo: its results, and a slide where you type text and the model guesses the next letter. |
| **Part 2 · [Memory with a door](slides/lstm-gru.html)** | 24 | Why plain RNNs forget, where the LSTM and GRU came from, how their gates work (cell diagrams, step-by-step equations), why blame survives along the LSTM's belt, measured results from the demo, and which cell to choose. |
| **Part 3 · [Read first, then write](slides/seq2seq.html)** | 23 | Sequence to sequence: why translation needs an encoder and a decoder, the context vector, start and end tokens, training end to end, teacher forcing, greedy vs beam search, and the fixed-size bottleneck. Then the date-translation demo, live: step through the encoder, c and the decoder; greedy vs beam on inputs where they disagree; and the bottleneck, measured. Ends with attention as the fix. |
| **Part 4 · [Look back while you write](slides/attention.html)** | 20 | Attention: keep every encoder memory, then at each decoder step score them, soften the scores with softmax and mix them into a fresh context. A softmax playground, a step-through of “the black cat”, why attention gives blame a short path back, and alignment maps. Then the date translator with attention, live: watch where it looks for each character, and four dates at once with no bottleneck. Ends with what attention doesn't fix and why the Transformer dropped the RNN. |
| **Part 5 · [Every word looks at every word](slides/self-attention.html)** | 22 | Self-attention: drop the RNN and let the sentence attend to itself. Queries, keys and values; the scaled dot product, with a √d<sub>k</sub> playground; one row worked by hand; what “it” refers to; the whole sentence as three matrix multiplies. Then the catch, live: shuffle the words and the outputs just shuffle, fixed by adding positions. Multi-head attention with a heads explorer, masked attention for writing, one Transformer block, and the RNN vs self-attention trade. Ends with a guide to its in-slide experiments and where trained self-attention runs (`transformer.py`, Part 6). No demo script of its own. |
| **Part 6 · [The whole machine](slides/transformer.html)** | 23 | The full encoder–decoder Transformer: embeddings and positions, stacked blocks, the feed-forward sublayer, residual connections, layer norm, the decoder's masked self-attention and cross-attention (Part 4's attention again), and the output layer. Training in one masked pass vs writing one character at a time, and measured training speed. Then the date translator as a Transformer, live: one encoder block step by step on real numbers, all eight cross-attention maps, where it fails on unseen years, and four dates at once. Ends with the three families: encoder–decoder, decoder-only (GPT), encoder-only (BERT). |
| **Part 7 · [Fill in the blanks](slides/bert.html)** | 20 | BERT: pretrain once, fine-tune many times; from word2vec and ELMo to BERT; the model as Part 6's encoder made big; WordPiece tokens, [CLS], [SEP] and segments; why an unmasked encoder can't learn by predicting the next word; the masked language model (15%, 80/10/10) and next sentence prediction. Then the real `bert-base-uncased`: guess a hidden word from both sides vs GPT-2 from the left only, “bank” by a river vs at a bank pulling apart block by block, all 144 attention heads, and a fine-tune on movie reviews (64% with no pretraining, 83% frozen, 88.5% fine-tuned). Ends with BERT vs GPT. |

Several slides are interactive: sliders for gates, a "you are the gates" puzzle, a blame race between a plain RNN and an LSTM, in Parts 3 and 4 the trained date translators running in the browser, including a live attention map, in Part 5 a √d<sub>k</sub> playground, a word-order test on a real self-attention layer and a heads explorer, and in Part 6 a layer-norm slider and the trained Transformer running in the browser, and in Part 7 BERT's real WordPiece tokenizer and explorers for its guesses, word vectors and attention heads.

Keys: **→ / ←** to move, **S** for speaker notes (each slide has notes for the presenter), **Esc** for an overview of all slides, **F** for full screen.

## Demo

Two scripts train a real model on about 1.1M characters of Shakespeare and let you test it, ask it for guesses and let it write. A third trains a sequence-to-sequence model that rewrites dates, a fourth adds attention to it, and a fifth rebuilds it as a Transformer. Each trains on a laptop CPU in about 25 seconds to 3 minutes (the multi-date experiments take longer). A sixth, `bert.py`, downloads a pretrained BERT instead of training one. Part 5 has no script: its experiments run inside the slides.

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

### `transformer.py`: the date translator, as a Transformer (Parts 5 and 6)

The same task, data and exam again, with no RNN anywhere: Part 5's self-attention and sine-wave positions in the encoder, masked self-attention and cross-attention in the decoder. d = 64, 4 heads, 2 blocks each side, 238,734 weights.

```sh
uv run demo/transformer.py train                                # ~3 min
uv run demo/transformer.py test
uv run demo/transformer.py translate "23 Oct 2026" --details    # cross-attention map, last block, all heads averaged
uv run demo/transformer.py translate "23 Oct 2026" --details --layer 2 --head 3
uv run demo/transformer.py speed                                # one training round: GRU + attention vs Transformer
uv run demo/transformer.py export                               # put the weights into the Part 6 slides
```

From one run (seed 0): 100% on unseen days, but 84.9% on unseen years (Part 4: 99.8%); its misses copy a familiar `20` instead of the `9` in `2091`. With `--max-dates 4 --steps 8000` (~20 min) it gets 100%, 99.8%, 99.7% and 98.4% for 1 to 4 dates.

Part 5 has no script of its own. Its experiments run inside the slides, on random or illustrative weights; the trained self-attention is `MultiHeadAttention` and `EncoderBlock` in `transformer.py`. `--details` prints the decoder's cross-attention, not the encoder's self-attention.

### `bert.py`: open up a pretrained BERT (Part 7)

No training from scratch: it downloads `bert-base-uncased` (~440 MB, once) and GPT-2 for one comparison, through Hugging Face's `transformers`.

```sh
uv run demo/bert.py tokens "Transformers revolutionised embeddings"     # WordPiece: em + ##bed + ##ding + ##s
uv run demo/bert.py fill "the man went to the [MASK] to buy a gallon of milk ." --left-only
uv run demo/bert.py senses bank                                        # one word, six sentences, 13 layers
uv run demo/bert.py attention "the animal didn't cross the street because it was too tired ." --layer 7 --head 11
uv run demo/bert.py nsp "the man went to the store ." "he bought a gallon of milk ."
uv run demo/bert.py finetune                                           # SST-2 movie reviews, ~3 min
uv run demo/bert.py classify "not bad at all ." "not good at all ."
uv run demo/bert.py export                                             # put the examples into the Part 7 slides
```

From one run: on 2,000 SST-2 training sentences, word vectors learned from scratch get 64.0% of the 872 dev sentences right, frozen BERT with one new layer 83.0%, and BERT fine-tuned end to end 88.5%.

[demo/README.md](demo/README.md) has the full walkthrough: the train / validation / test split, what one training round does, sample output from every command, and which lesson each part of the code matches.

Trained models (`demo/*.pt`, `demo/bert-sst2*/`) aren't in the repo; `train` and `finetune` create them.

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
