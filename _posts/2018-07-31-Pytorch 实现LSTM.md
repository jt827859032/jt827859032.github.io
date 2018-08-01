---
title: Pytorch 实现LSTM
date: 2018-07-31 01:03:53
categories:
- Pytorch
tags:
- Deep Learning
- LSTM
---

Reference: [Andre Holzner](https://medium.com/@andre.holzner/lstm-cells-in-pytorch-fab924a78b1c)

![LSTM](http://p6a2eqn18.bkt.clouddn.com/8.png)

The yellow boxes correspond to matrix multiplication followed by non-linearities. W represent the weight matrices, the bias terms b have been omitted for simplicity. The mathematical symbols used in this diagram correspond to those used in PyTorch’s documentation of [torch.nn.LSTM](http://pytorch.org/docs/master/nn.html#torch.nn.LSTM):

-  **external input** (e.g. from training data) at time t
- h(t-1)/h(t): the **hidden state** at times t-1 (‘input’) or t (‘output’). Despite its name, this is also used as **output** or used as input for a next layer of LSTM cells (for multi-layer networks)
- c(t-1)/c(t): the **‘cell state’** or **‘memory’** at times t-1 and t
- f(t): the result of the **forget gate**. For values close to zero the cell will ‘forget’ its memories c(t-1) from the past, for values close to one it will remember its history.
- i(t): the result of the **input gate**, determining how important the (transformed) new external input is.
- g(t): the result of the **cell gate**, a non-linear transformation of the new external input x(t)
- o(t): the result of the **output gate** which controls how much of the new cell state c(t) should go to the output (and the hidden state)

It is also instructive to look at the implementation of [torch.nn._functions.rnn.LSTMCell](https://github.com/pytorch/pytorch/blob/c62490bf597ec93f308a8b0108522aa9b40701d9/torch/nn/_functions/rnn.py#L23) :

```python
def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    if input.is_cuda:
        ...
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate     = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate   = F.tanh(cellgate)
    outgate    = F.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)
    return hy, cy
```

The second argument in fact is expected to be a tuple of:

(hidden state at time t-1, cell/memory state at time t-1) and the return value is of the same format but for time t.