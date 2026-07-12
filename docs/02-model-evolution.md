# Model Evolution: From State Compression to External Memory

## RNN: temporal state

RNNs made histories differentiable. They enabled temporal credit assignment but compressed a growing past into a fixed hidden state and required sequential computation. LSTM and GRU gates improved memory preservation without eliminating the bottleneck.

## Attention and Transformers: relational context

Attention replaced a single compressed history with content-dependent access to elements in context. Transformers removed recurrence from the central architecture, improved parallelism, and made preference-conditioned multi-task and Pareto models easier to scale.

## Early-fusion multimodality: common token space

Chameleon-style models tokenize text and images for a shared autoregressive model. In optimization terms, this expands the state: objectives can depend jointly on language, visual evidence, and structured observations. It also introduces modality balance, token allocation, and safety trade-offs.

## Vector augmentation: non-parametric memory

Embedding retrieval overlays relevant external evidence onto model context. The model no longer needs every fact encoded in its parameters. The retrieval system becomes part of the optimization target: chunk size, top-k, reranking, relevance, latency, cost, privacy, and prompt budget interact.

## Agentic loop: adaptive experiment selection

A mature system can propose candidates, call evaluators, retrieve prior results, estimate Pareto improvement, and select the next experiment. This resembles Bayesian or evolutionary optimization with a learned policy and tool access. The gain is adaptivity; the risks are unstable feedback, proxy gaming, and inadequate human control.

## Summary

The architectural progression expands the information available at decision time:

```text
RNN          = compressed temporal memory
Transformer  = globally related context
Multimodal   = unified cross-modal context
Vector/RAG   = retrieved external context
Agentic      = iterative context, tools, and feedback
```

Non-convexity remains. What improves is the system's capacity to represent the problem, condition on preferences, propose candidates, and evaluate trade-offs.

