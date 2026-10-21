# Recurrent Neural Networks Resources

## Knowledge

- [Article: "The Unreasonable Effectiveness of Recurrent Neural Networks" by Andrej Karpathy](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  The classic intuition-first intro: sequences of vectors, the step function, and char-level text generation demos. Use for: why RNNs matter, task shapes (the one-to-many/many-to-one diagram), the core step equation.
- [Article: "Understanding LSTM Networks" by Christopher Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  The best visual explanation of unrolling, long-term dependencies and LSTM gates. Use for: unrolled diagrams, the long-term dependency problem, every LSTM lesson.
- [Book: _Dive into Deep Learning_, Ch. 9 "Recurrent Neural Networks" (Zhang, Lipton, Li, Smola)](https://d2l.ai/chapter_recurrent-neural-networks/)
  Rigorous open textbook with equations and shapes. Use for: the precise hidden-state equation (§9.4), BPTT (§9.7), Ch. 10 for LSTM/GRU, and §10.7 for encoder–decoder / seq2seq.
- [Book: _Deep Learning_, Ch. 10 "Sequence Modeling: Recurrent and Recursive Nets" (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/contents/rnn.html)
  The authoritative reference text. Use for: formal definitions and interview-grade depth on BPTT and long-term dependencies.

- [Paper: "Sequence to Sequence Learning with Neural Networks" by Sutskever, Vinyals & Le (2014)](https://arxiv.org/abs/1409.3215)
  The original LSTM encoder–decoder for translation. Use for: seq2seq history, the fixed-size context vector, the source-reversal trick.
- [Paper: "Neural Machine Translation by Jointly Learning to Align and Translate" by Bahdanau, Cho & Bengio (2014)](https://arxiv.org/abs/1409.0473)
  Names the fixed-length context vector as a bottleneck and introduces attention. Use for: the bridge from RNNs to Transformers.

- [Paper: "On the difficulty of training Recurrent Neural Networks" by Pascanu, Mikolov & Bengio (2012; ICML 2013)](https://arxiv.org/abs/1211.5063)
  The standard analysis of vanishing/exploding gradients in RNNs; introduces gradient norm clipping. Use for: interview-depth answers on why gradients vanish and how clipping works.

## Wisdom (Communities)

- [r/learnmachinelearning](https://www.reddit.com/r/learnmachinelearning/)
  Large, beginner-friendly community. Use for: sanity-checking your explanations and asking "is this how you'd explain X in an interview?".
- [Cross Validated (stats.stackexchange.com)](https://stats.stackexchange.com/questions/tagged/rnn)
  Moderated Q&A with the `rnn` tag. Use for: precise conceptual questions (e.g. why gradients vanish).

## Gaps
- A short, high-trust video explainer (StatQuest's RNN video is a candidate; URL not yet verified).
- Curated bank of real interview questions on RNNs.
