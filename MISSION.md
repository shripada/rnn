# Mission: Recurrent Neural Networks

## Why
Be able to explain RNNs confidently in an interview or exam setting: what they are, why they exist, how they're trained, and why LSTMs/GRUs (and later Transformers) replaced vanilla RNNs.

## Success looks like
- Draw an unrolled RNN on a whiteboard and explain the hidden state without notes
- Write the hidden-state update equation and state the shape of every matrix
- Explain parameter sharing and why an RNN handles variable-length input
- Explain backpropagation through time and why gradients vanish/explode
- Explain how LSTM gates fix this, and contrast with GRUs and Transformers
- Name the main sequence task shapes (one-to-many, many-to-one, many-to-many) with an example of each
- Explain self-attention with queries, keys and values, why scores are divided by √d, why position information is needed, and what multi-head attention adds

## Constraints
- ~45 minute sessions
- Concepts + intuition first; minimal code
- Backprop is understood only conceptually, so math is built up gently

## Out of scope
- Implementing RNNs from scratch or training models in PyTorch (for now)
- Transformer internals beyond self-attention, multi-head attention, position information and a one-block overview (no training recipes, tokenisers, normalisation variants or scaling laws)
