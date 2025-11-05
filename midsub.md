Got it — I converted all math into Markdown-friendly LaTeX (inline with `$...$` and display with `$$...$$`) so it will render correctly in Markdown viewers (GitHub, VS Code, Jupyter, etc.). Here’s the full document:

---

# Proposed Methodology

We present **CEGE: Conversational Emotion Graph Evolution**, a novel architecture that models conversations as dynamically evolving temporal graphs for Emotion Recognition in Conversation (ERC).
Our approach is built from three tightly integrated components: **utterance encoding**, **dynamic temporal graph construction**, and **emotionally-aware classification**. Each is designed to capture the evolving nature of emotions and speaker relationships as dialogue unfolds.

---

## Problem Definition

Let there be $M$ speakers/parties $ p_1, p_2, \ldots, p_M $ in a conversation.
The task is to predict the **emotion labels** (*happy, sad, neutral, angry, excited, frustrated, disgust,* and *fear*) of the constituent utterances $u_1, u_2, \ldots, u_N$, where utterance $u_i$ is spoken by speaker $p_{s(u_i)}$, with $s$ being the mapping between utterance and its corresponding speaker.

We also represent $u_i \in \mathbb{R}^{D_m}$ as the feature representation of the utterance, obtained using the feature extraction process described below.

---

## Utterance Encoding

We employ **convolutional neural networks (CNNs)** for textual feature extraction. Each utterance is passed through three distinct convolution filters of sizes **3**, **4**, and **5**, each with **50 feature maps**. Outputs are then subjected to **max-pooling** followed by **ReLU** activation.

These activations are concatenated and fed into a **100-dimensional dense layer**, which forms the textual utterance representation. This network is trained at the utterance level using the emotion labels.

---

## Sequential Context Encoding

Since conversations are sequential by nature, contextual information flows along that sequence. We feed the conversation into a **bidirectional GRU** to capture this contextual information:

$$
g_i = \overleftrightarrow{\mathrm{GRU}}*S\bigl(g*{i(\pm 1)}, u_i\bigr), \quad \text{for } i = 1,2,\ldots,N,
$$

where $u_i$ and $g_i$ are context-independent and sequential context-aware utterance representations, respectively.

Because the utterances are encoded irrespective of the speaker, this initial encoding scheme is **speaker-agnostic**.

---

## Graph Representation and Temporal Memory

A conversation with $N$ utterances is represented as a directed graph:

$$
\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R}, \mathcal{W})
$$

where:

* $\mathcal{V} = {v_1, v_2, \dots, v_N}$ is the set of nodes, each node corresponding to an utterance.
* $\mathcal{E}$ is the set of directed edges, where each edge $r_{ij}$ connects node $v_j$ (source) to node $v_i$ (target).
* $\mathcal{R}$ is the set of relation types (speaker dependencies, temporal order, learned interaction patterns).
* $\mathcal{W} = {\alpha_{ij}}$ is the set of edge weights, with $\alpha_{ij} \in [0,1]$ quantifying the influence of utterance $j$ on $i$.

### Temporal Memory Modules

Each node $v_i$ maintains a temporal memory state $\mathbf{m}_i^t$ at time $t$, summarizing evolving emotional and contextual information:

$$
\mathbf{m}_i^t = \mathrm{LSTM}\bigl(\mathbf{m}_i^{,t-1}, \mathbf{g}_i, \text{context}^t\bigr),
$$

where:

* $\mathbf{m}_i^{,t-1}$ is the previous memory state for node $i$.
* $\mathbf{g}_i$ is the context-aware embedding of utterance $i$.
* $\text{context}^t$ includes:

  * $\mathbf{Q}_s^t$: the speaker state for speaker $s$ at time $t$, capturing emotional and interactional behavior (for utterance $i$, the relevant speaker is $s(i)$).
  * $\mathbf{C}^t$: the global conversation state at time $t$, encoding overall topic flow and emotional climate.

### Graph Evolution

At every new utterance (time $t$), the graph is updated as follows:

1. **Temporal state change**:
   $$
   \Delta_t = f_{\text{analyze}}\bigl(\mathbf{M}^{,t-1}, \mathbf{g}_t, \mathbf{Q}_s^{,t-1}\bigr),
   $$
   where $\mathbf{M}^{,t-1}$ is the set of all node memories at time $t-1$.

2. **Edge weights update**:
   $$
   \alpha_{ij}^t = f_{\text{temporal}}\bigl(\alpha_{ij}^{,t-1}, \Delta_t, \mathbf{Q}*{s(i)}^{,t-1}, \mathbf{Q}*{s(j)}^{,t-1}\bigr),
   $$
   where $f_{\text{temporal}}$ is a learned function (for example, a neural network or an attention mechanism).

3. **Edge pruning and addition**:

   * Remove edges with $\alpha_{ij}^t < \tau_{\text{remove}}$.
   * Add new edges if candidate scores exceed $\tau_{\text{create}}$, enabling capture of emerging long-range or non-local relationships.

4. **Relation type adaptation**: update the set of relation types $\mathcal{R}^t$ dynamically based on learned interaction patterns (e.g., emotional influence, topic shift, agreement/disagreement).

---

## Temporal Graph Convolution and Attention

Node features are passed through two temporal GCN layers at each time $t$.

### Layer 1

$$
\mathbf{h}*i^{(1),t} = \sigma\Bigg(
\sum*{r \in R^t}\sum_{j \in \mathcal{N}*i^r}
\frac{\alpha*{ij}^t}{c_{i,r}}, W_r^{(1)} \bigl[\mathbf{g}_j ,|, \mathbf{m}_j^t\bigr]
;+; W_0^{(1)} \bigl[\mathbf{g}_i ,|, \mathbf{m}_i^t\bigr]
\Bigg),
$$

where:

* $\mathbf{h}_i^{(1),t}$ is the first-layer hidden feature for node $i$ at time $t$.
* $\sigma$ is the activation function (typically $\mathrm{ReLU}$).
* $R^t$ is the set of relation types at time $t$.
* $\mathcal{N}_i^r$ is the set of neighbors of node $i$ with relation $r$.
* $\alpha_{ij}^t$ is the edge weight from node $j$ to node $i$ at time $t$.
* $c_{i,r}$ is a normalization constant (e.g., number of neighbors of type $r$).
* $W_r^{(1)}$ is a learnable weight matrix for relation $r$ in layer 1.
* $[\mathbf{g}_j ,|, \mathbf{m}_j^t]$ denotes concatenation.
* $W_0^{(1)}$ is a learnable self-connection weight matrix.

### Layer 2

$$
\mathbf{h}*i^{(2),t} = \sigma!\left(
\sum*{j \in \mathcal{N}_i} W^{(2)} \mathbf{h}_j^{(1),t}
;+; W_0^{(2)} \mathbf{h}_i^{(1),t}
\right),
$$

where $W^{(2)}$ and $W_0^{(2)}$ are learnable weight matrices for neighbor aggregation and self-connections.

### Temporal Attention

To focus on relevant history, attention weights are computed:

$$
\boldsymbol{\beta}_i^t = \mathrm{softmax}!\left( \bigl(\mathbf{h}*i^{(2),t}\bigr)^\top W*\beta ;[\mathbf{m}*1^t, \ldots, \mathbf{m}*{i-1}^t] \right),
$$

where $W_\beta$ is a learnable weight matrix and $[\mathbf{m}*1^t, \ldots, \mathbf{m}*{i-1}^t]$ are the memory states of previous utterances.

Temporal decay is applied:

$$
\beta_{ik}^t \leftarrow \beta_{ik}^t \cdot \exp\bigl(-\lambda_{\text{decay}} \cdot (t-k)\bigr),
$$

where $\lambda_{\text{decay}}$ is a hyperparameter controlling decay rate.

The context vector is:

$$
\mathbf{c}*i^t = \sum*{k=1}^{i-1} \beta_{ik}^t \cdot \mathbf{m}_k^t,
$$

which summarizes the most relevant historical context for node $i$ at time $t$.

---

## Training and Inference

The total loss minimized during training is:

$$
\mathcal{L} = \mathcal{L}*{\text{CE}} ;+; \lambda*{\text{reg}} \lVert \theta \rVert_2^2
;+; \lambda_{\text{temp}} \sum_i \bigl\lVert \mathbf{m}_i^t - \operatorname{stop_grad}(\mathbf{m}_i^{,t-1}) \bigr\rVert_2^2,
$$

where:

* $\mathcal{L}_{\text{CE}}$ is the **cross-entropy loss** for emotion classification.
* $\lambda_{\text{reg}} \lVert \theta \rVert_2^2$ is the **L2 regularization** term for model parameters $\theta$.
* $\lambda_{\text{temp}} \sum_i \lVert \mathbf{m}_i^t - \operatorname{stop_grad}(\mathbf{m}_i^{,t-1}) \rVert_2^2$ is the **temporal consistency loss**, encouraging smooth memory evolution; `stop_grad` treats the previous state as a fixed reference during backprop.

**Inference:** At test time, the model processes each conversation sequentially, updating the dynamic graph and temporal memory states per utterance. Emotion predictions are produced for each utterance using the most up-to-date context and memory information.

---

If you want, I can:

* produce a **GitHub-flavored Markdown (.md)** file for download, or
* convert the math to images (SVG/PNG) if your Markdown renderer does **not** support LaTeX, or
* tweak inline vs display styles (e.g., use `$$...$$` for all displayed equations).

Which would you like next?
