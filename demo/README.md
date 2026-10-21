# Demo: a character-level RNN on S

## The problem

**Given the text so far, predict the next character.** For example, `"ROME"` → `"O"`, and `"KING RICHAR"` → `"D"`.

This is *language modelling*, one character at a time: the task behind autocomplete. It needs an RNN because the answer depends on characters **further back** than the last one. After `"ROME"`, a guesser that only sees the last character (`E`) picks `N`, `S` or `R`, never `O`. The model needs **memory** (the hidden state) and has to handle **any length** of text (Lesson 1).

## The data: train, val, test

About 1.1M characters of Shakespeare plays, downloaded on first run, split by position:

| part | share | used by | purpose |
|---|---|---|---|
| **TRAIN** | first 80% | `train` | the text the model learns from |
| **VAL** | next 10% | `train` | a check-up during training: is it learning, or just memorising? |
| **TEST** | last 10% | `test` | the final exam, on scenes the model has never seen |

## The four commands

```sh
uv run demo/char_rnn.py train                      # 1. train and save to demo/model.pt (~25 s)
uv run demo/char_rnn.py test                       # 2. grade it on the unseen TEST text
uv run demo/char_rnn.py predict "KING RICHAR"      # 3. top guesses for the next character
uv run demo/char_rnn.py generate "ROMEO:" --n 300  # 4. let it write
```

(Or `source .venv/bin/activate` and use `python demo/char_rnn.py …`.)

### 1. `train`: what one training round does
Repeated 2000 times:
1. **Guess.** Take 64 random 100-character snippets of TRAIN text. The RNN reads each one a character at a time and guesses the next character at every position.
2. **Check.** Compare every guess with the real next character. The *loss* measures how wrong the guesses were.
3. **Work out the blame.** Backpropagation through time (Lesson 3) computes which way to nudge each weight.
4. **Safety.** Gradient clipping caps a huge gradient (Lesson 3).
5. **Adjust.** Nudge every weight a little in that direction.

The loss falls from 4.17 (random guessing) to about 1.5. At the end it saves the weights to `demo/model.pt`.

### 2. `test`: grading on unseen text
From a real run:
```
How often is the top guess for the next character exactly right?
  random guessing (1 in 65):      1.5%
  last-char-only guesser:        26.5%   (no memory)
  your trained RNN:              47.2%   (loss 1.85)
```
It also prints a few test lines with the RNN's guess under each character (`·` = guessed right).

### 3. `predict`: try your own text
```
Text so far: 'ROME'
  trained RNN:                        O 87%, R 8%, E 1%, : 1%, S 1%
  last-char-only guesser (no memory): N 21%, S 12%, R 12%, T 9%, : 8%
```

### 4. `generate`: let it write
It reads your prompt, picks a likely next character, feeds it back in, and repeats. The output looks like a play (names in capitals, line breaks, mostly real words) but doesn't make sense. A small model trained for 25 seconds learns spelling and layout, not meaning.

## Swapping the cell: LSTM and GRU (Lessons 4–5)

Every command takes `--cell rnn | lstm | gru` (default `rnn`). Each cell type saves to its own file (`model.pt`, `model-lstm.pt`, `model-gru.pt`), so you can train all three and compare them:

```sh
uv run demo/char_rnn.py train --cell lstm --details   # ~80 s
uv run demo/char_rnn.py test  --cell lstm
uv run demo/char_rnn.py train --cell gru --details    # ~65 s
uv run demo/char_rnn.py generate "ROMEO:" --cell gru
```

From a real run (hidden size 256, 2000 rounds, same seed):

| | vanilla RNN | LSTM | GRU |
|---|---|---|---|
| weights | 99,393 | 347,457 | 264,769 |
| training time | 24 s | 80 s | 64 s |
| test: next char right | 47.1% | 50.0% | 50.3% |
| probe: gradient 40 steps back | 1.1e−5 | 2.8e−4 | 1.4e−3 |
| probe: gradient 99 steps back | 2.3e−9 | 2.7e−6 | 6.1e−7 |

The gated cells keep far more gradient alive far back and score a little better. One run doesn't say which of LSTM and GRU is better; studies find them about equal.

## Which lesson each part matches

| Part | Lesson |
|---|---|
| The model is `h_t = tanh(W_hh·h_{t−1} + W_xh·x_t + b)`; same weights every step | 1 |
| `train` puts a target at every step: synced many-to-many | 2 |
| `predict` reads everything and gives one answer: many-to-one | 2 |
| `generate` feeds each output back in: one-to-many | 2 |
| BPTT, gradient clipping, `--seq-len` = truncated BPTT window | 3 |
| `train --details`: gradient norms plus a probe showing the gradient vanishing over time | 3 |
| `--cell lstm`: gates and an additive cell state, so the probe's gradient survives further back | 4 |
| `--cell gru`: two gates, one state, about 3/4 of the LSTM's weights | 5 |

Based on Karpathy's [char-rnn](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).

## Predicting words instead of characters: `word_rnn.py`

The same task one level up: **given the words so far, predict the next word.** Same model, same training loop, same four commands; each RNN step now reads a whole word.

```sh
uv run demo/word_rnn.py train                          # LSTM by default, ~70 s
uv run demo/word_rnn.py test
uv run demo/word_rnn.py predict "good morrow, my"
uv run demo/word_rnn.py generate "romeo:\n" --n 80
uv run demo/word_rnn.py neighbours lord king she     # words with the closest learned embeddings
```

What changes:

- **Tokens.** Text is lowercased and split into words, punctuation marks and newlines: `"ROMEO:\nIs she dead?"` → `romeo : ↵ is she dead ?`
- **Vocabulary.** ~4,300 words instead of 65 characters. Words seen fewer than 3 times in TRAIN become `<unk>` (unknown); the rest cover 96% of the text.
- **Embedding.** A learned table `E` with one row of 128 numbers per word. Reading a word means looking up its row; the RNN cell then multiplies it by its own input matrix `W_ih`. So Lesson 1's `W_xh` is split in two (`W_xh = W_ih · Eᵀ`), which squeezes every word through 128 numbers. `E` is learned from scratch, not pretrained (no word2vec). It starts small (`std=0.1`): at PyTorch's default size (~1) the rows barely moved during training and stayed near-random.
- **Keeping the best weights.** With many more weights per word of text, the model starts memorising TRAIN after ~600 rounds (train loss falls, val loss rises). `train` keeps the weights from the round with the lowest val loss.

From a real run (embedding 128, hidden 256, `test` on unseen text):

```
How often is the top guess for the next token exactly right?
                                 all tokens  words only
  random guessing (1 in 4,299)        0.02%       0.02%
  last-word-only guesser              23.5%        5.3%   (no memory)
  your trained LSTM                   29.5%        8.4%

  right answer in the model's top 5: 47.9%
  loss 4.08  ->  e^loss ≈ 59 words it's still choosing between (out of 4,299)
```

```
Words so far: good morrow , my
  trained LSTM:                       lord 44%, gracious 5%, <unk> 5%, liege 4%, ...
  last-word-only guesser (no memory): lord 12%, <unk> 6%, heart 3%, life 2%, soul 2%
```

Exact-match accuracy looks low because each guess is 1 of ~4,300, and many next words are genuinely open. The top-5 rate and the memory gap over the last-word guesser are the better signals. `--cell rnn | lstm | gru` works here too (one run: words right 7.8% / 8.4% / 8.1%).

`neighbours` shows what the embedding learned, by cosine similarity (1 = same direction):
```
she:  he 0.81, it 0.73, there 0.59, they 0.56, hers 0.55
king: prince 0.56, favour'd 0.55, brother 0.52, moved 0.49, duke 0.48, lord 0.47
lord: sir 0.60, master 0.58, lady 0.55, softly 0.53, ...
```
Nobody told it what words mean. Words end up close only because they're used in similar places. Common words get clean neighbours; rare ones stay noisy, since they're seen too rarely to move far.
