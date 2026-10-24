# Demo: a character-level RNN on Shakespeare

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

## Sequence to sequence: `seq2seq.py` (Part 3)

### The problem

**Read a date written the way people write it, and write it as `YYYY-MM-DD`.**

```
23 Oct 2026                                       -> 2026-10-23
Friday, 23 October 2026                           -> 2026-10-23
the twenty-third of October, twenty twenty-six    -> 2026-10-23
```

A tiny translation task. Input and output have different lengths, different alphabets and a different order (the day is read first and written last), so a synced many-to-many RNN can't do it. It needs Part 3's shape: an **encoder** that reads the whole input, and a **decoder** that writes from what the encoder left behind.

### The model

- **Encoder:** a GRU reads the input one character at a time (one-hot, like `char_rnn.py`). Its last hidden state is the **context c**: 128 numbers.
- **Decoder:** a second GRU, with its own weights, starts from `c`. It's fed the start token `⟨s⟩`, writes a character, is fed that character, and so on until it writes the end token `⟨/s⟩`.
- 133,134 weights in all. Both halves are trained together: one loss at the decoder, and blame flows back through `c` into the encoder.

### The data

Generated in code: random dates in six formats (`23 Oct 2026`, `23 October 2026`, `October 23, 2026`, `Oct 23 2026`, `Friday, 23 October 2026`, and spelled out), 15% all lower case and 15% all upper case.

| part | dates | used by |
|---|---|---|
| **TRAIN** | 1900–2079, except every 10th day | `train` |
| **TEST (days)** | every 10th day of 1900–2079 | `test` |
| **TEST (years)** | every day of 2080–2099 | `test` |

`23/10/2026` is never in training. The encoder can read `/` and `.`, but their weights were never trained.

### The commands

```sh
uv run demo/seq2seq.py train                                # ~70 s
uv run demo/seq2seq.py test
uv run demo/seq2seq.py translate "23 Oct 2026" --details    # top 3 guesses at every decoder step
uv run demo/seq2seq.py translate "SEP 13 2088" --beam 3     # greedy vs beam search
uv run demo/seq2seq.py export                               # write the weights into slides/seq2seq.html
```

Every command takes `--cell rnn | lstm | gru` (default `gru`), `--hidden N` (the size of `c`, default 128) and `--max-dates N` (default 1). Each combination saves to its own file, e.g. `seq2seq-gru-128.pt` or `seq2seq-gru-128-x4.pt`.

`train` runs 4,000 rounds of 128 random dates. The learning rate falls in a straight line to a tenth of its start; without it, some seeds kept bouncing around near the answer (one stopped at 74% after 2,500 rounds). With it, seeds 0–3 all reached 100% on the validation dates.

### `test`: what it learned, and what it didn't

From a real run (seed 0):
```
Whole date exactly right: 100.0%          (39,444 inputs from 6,574 days it never saw)

Years it never saw (2080-2099), all formats: 12.1%
  Tuesday, 13 March 2091    -> 2001-03-13   should be 2091-03-13
  August 8, 2082            -> 2032-08-08   should be 2082-08-08
```
Perfect on new days, poor on new years. The decoder has never once written `208` or `209`, so it writes a year it knows. It learned patterns, not the rule "copy the year".

### `translate`: greedy vs beam search

```
Input: 'the twenty-third of May, twenty ninety-two'
  greedy:  2022-05-23                    (whole answer 20%)
  beam search, width 3:
    1. 2092-05-23     31.1%              <- right
    2. 2022-05-23     20.0%
    3. 2020-08-23     12.2%
```
Greedy commits to the likeliest character after `20` (a `2`) and is stuck with it. Beam search keeps `209…` alive and finds that the whole answer is more likely. On easy inputs (`23 Oct 2026`, 95%) the two agree. Inputs it has never seen give confident nonsense: `23/10/2026` → `2040-04-22`, and `2026-10-23` → an empty answer.

### The bottleneck: `--max-dates 4`

A second model reads 1 to 4 dates at once (`23 Oct 2026; May 5, 1990`) and writes them all (`2026-10-23;1990-05-05`), through the same `c`.

```sh
uv run demo/seq2seq.py train --max-dates 4     # ~5 min
uv run demo/seq2seq.py test  --max-dates 4
```

Whole answer right, on dates it never saw:

| model | 1 date | 2 dates | 3 dates | 4 dates |
|---|---|---|---|---|
| c = 128 | 99.3% | 0.6% | 0.0% | 0.0% |
| c = 128, trained 3× longer (`--steps 12000`) | 100% | 3.7% | 0.0% | 0.0% |
| c = 512 (`--hidden 512`) | 100% | 20.2% | 0.1% | 0.0% |

It always gets the **first** date right, then garbles the rest (`1990-05-05` → `1904-05-06`). Training longer barely helps, and a bigger `c` does, so `c` really is a limit. Even 512 numbers only get two dates right a fifth of the time, though: part of the difficulty is the decoder learning to pull a second date back out of one summary. Attention (next lesson) removes the single summary altogether.

### Teacher forcing: `--no-teacher-forcing`

Feeds the decoder its own guesses during training instead of the right previous character. On this task it makes no consistent difference: over three seeds, either version was sometimes faster to learn, and the seed mattered more than the setting. Teacher forcing earns its keep on long outputs with big vocabularies, which this task doesn't have.

### Which slide each part matches (Part 3)

| Part | Slide |
|---|---|
| `Seq2Seq.encode`: read everything, keep only the last `h` | The encoder is a many-to-one RNN |
| `decode_step`, starting from `c`, fed `⟨s⟩` | The decoder is our Shakespeare model, with a head start |
| One loss, `loss.backward()` through decoder, `c` and encoder | Encoder and decoder learn together |
| `forward(..., teacher_forcing=True)` | If the decoder guesses wrong, what do we feed it next? |
| `greedy` and `beam_search` | Don't lock in the first word too early |
| `--max-dates 4` | Squeeze the whole sentence into c |

## Attention: `attention.py` (Part 4)

### The problem

The same as `seq2seq.py`: read a date written any of six ways, write it as `YYYY-MM-DD`. Same data, same train / test split, same exam. The only change is the model.

### The model

- **Encoder:** the same GRU, but we keep its memory after **every** character, `h_1 … h_T`, not just the last one.
- **Decoder:** the same GRU, started from `h_T` as before. At every step, before writing, it looks back:
  1. **score** every `h_j`: `e_j = v · tanh(W_s s + W_h h_j)`, how useful is position `j` for what I'm about to write?
  2. **soften** the scores with softmax into weights `a_j` that add up to 1;
  3. **mix**: `c_t = Σ a_j h_j`, a fresh context for this step. It's fed into the decoder with the previous character, and into the output layer.
- 200,590 weights; attention (`W_s`, `W_h`, `v`) adds 16,512 of them. Data, greedy decoding and beam search are imported from `seq2seq.py` unchanged.

### The commands

```sh
uv run demo/attention.py train                                # ~2 min
uv run demo/attention.py test
uv run demo/attention.py translate "23 Oct 2026" --details    # where it looked for each character
uv run demo/attention.py export                               # write the weights into slides/attention.html
uv run demo/attention.py export --max-dates 4                 # and the 4-date model, for the bottleneck slide
```

Every command takes `--hidden N` (default 128) and `--max-dates N` (default 1), and saves to its own file: `attention-gru-128.pt`, `attention-gru-128-x4.pt`.

### `test`: the unseen years, fixed

From a real run (seed 0):
```
Whole date exactly right: 100.0%          (39,444 inputs from 6,574 days it never saw)

Years it never saw (2080-2099), all formats: 99.8%
  year written in digits ('2091'):             99.9%
  year spelled out ('twenty ninety-one'):      99.3%
```
Part 3 got 12% on the unseen years: it had never written `208x` or `209x`, so it wrote a year it knew. With attention, writing the third digit, the decoder looks at the input position that holds the `9` and passes it on. We didn't add attention for this; it's a side effect of being able to look. It isn't a perfect rule either: the few mistakes look like `2094` → `2994`.

### `translate --details`: the attention map

```
          23 Oct 2026
          ‾‾‾‾‾‾‾‾‾‾‾
     2 ←         ▓·     looked at '2' (#8) 74%
     0 ←         ░▒·    looked at '0' (#9) 53%
     2 ←           █    looked at '2' (#10) 90%
     6 ←            █   looked at '6' (#11) 99%
     - ←   ·       ··   looked at '6' (#11) 23%
     1 ←      ▒░·       looked at 'c' (#5) 55%
     0 ←      ▓··       looked at 'c' (#5) 70%
     - ←  ░░·   ·       looked at '3' (#2) 37%
     2 ←  █             looked at '2' (#1) 96%
     3 ←   █            looked at '3' (#2) 99%
  ⟨/s⟩ ←  ····  ·       looked at '2' (#1) 21%
```
One row per character written, one column per character read. It writes the year looking at the year, then jumps back to `Oct` for the month and to `23` for the day: the order flip, done by looking. Because the encoder reads left to right, it looks at the first letter that settles a word: the `c` of `Oct`, the `r` of `March` (not `May`), the `w` of `two` (not `three`).

`23/10/2026`, a format it never saw, comes out as `2026-12-23`: day and year copied right, but no training format writes the month as a number. Attention lets it copy; it doesn't teach it what a month number means.

### The bottleneck, gone: `--max-dates 4`

```sh
uv run demo/attention.py train --max-dates 4     # ~14 min (--steps 1000: ~3.5 min)
uv run demo/attention.py test  --max-dates 4
uv run demo/attention.py translate "23 Oct 2026; May 5, 1990" --max-dates 4 --details
```

Whole answer right, on dates neither model saw:

| model | 1 date | 2 dates | 3 dates | 4 dates |
|---|---|---|---|---|
| `seq2seq.py`, one c of 128 | 99.3% | 0.6% | 0.0% | 0.0% |
| `attention.py`, h of 128 | 100% | 100% | 100% | 100% |
| `attention.py`, `--steps 1000` | 99.5% | 99.4% | 98.5% | 97.9% |

The map for several dates shows one block per date, marching down the input: to write date 3, it looks at date 3. The price is time: every decoder step scores every input character, so the 4-date model trains about 3× slower than Part 3's.

### Which slide each part matches (Part 4)

| Part | Slide |
|---|---|
| `AttnSeq2Seq.encode` returns every `h_j` | Keep every memory, and look up the one you need |
| `AttnSeq2Seq.attend`: score, softmax, weighted sum | Score, soften, mix |
| `W_s`, `W_h`, `v` | A tiny network asks: "is this what I need?" |
| `decode_step`: `c_t` into the decoder and the head | What changed from Part 3 |
| `translate --details` | Watch where it looks |
| `--max-dates 4` | Four dates, no single c · More dates, no bottleneck |
