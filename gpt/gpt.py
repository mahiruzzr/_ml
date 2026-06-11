import os, math, random, sys

random.seed(42)

# ──────────────────────────────────────────────
# 1. Dataset — load names, one per line
# ──────────────────────────────────────────────
DATASET = sys.argv[1] if len(sys.argv) > 1 else 'input.txt'
if not os.path.exists(DATASET):
    print(f"Error: {DATASET} not found"); sys.exit(1)

docs = [l.strip() for l in open(DATASET, encoding='utf-8').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"dataset: {DATASET}, num docs: {len(docs)}")

# ──────────────────────────────────────────────
# 2. Tokenizer — character-level, 26 letters + BOS
# ──────────────────────────────────────────────
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)                     # special: Beginning Of Sequence
vocab_size = len(uchars) + 1          # 27 tokens total
print(f"vocab size: {vocab_size} (26 letters + BOS)")

# ──────────────────────────────────────────────
# 3. Autograd Engine — the Value class
#    Wraps a scalar, tracks computation graph,
#    and computes gradients via reverse-mode AD.
# ──────────────────────────────────────────────
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data              # forward value
        self.grad = 0                 # d(loss)/d(self), accumulated
        self._children = children     # nodes that produced this node
        self._local_grads = local_grads  # d(self)/d(child_i) for each child

    # ── arithmetic operators & their local gradients ──
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data + o.data, (self, o), (1, 1))
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data * o.data, (self, o), (o.data, self.data))
    def __pow__(self, n):
        return Value(self.data ** n, (self,), (n * self.data ** (n - 1),))
    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))
    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self):        return self * -1
    def __radd__(self, o):    return self + o
    def __sub__(self, o):     return self + (-o)
    def __rsub__(self, o):    return o + (-self)
    def __rmul__(self, o):    return self * o
    def __truediv__(self, o): return self * o ** -1
    def __rtruediv__(self, o):return o * self ** -1

    # ── backward: chain rule over topological order ──
    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children: build(c)
                topo.append(v)
        build(self)
        self.grad = 1           # d(loss)/d(loss) = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad

# ──────────────────────────────────────────────
# 4. Model Architecture — a mini GPT
# ──────────────────────────────────────────────
n_embd = 8          # embedding dimension
n_head = 2          # number of attention heads
n_layer = 1         # number of transformer layers
block_size = 16     # max context length
head_dim = n_embd // n_head  # 4

def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte':     matrix(vocab_size, n_embd),                    # token embeddings
    'wpe':     matrix(block_size, n_embd),                    # position embeddings
    'lm_head': matrix(vocab_size, n_embd),                    # output projection
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # query projection
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # key projection
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # value projection
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # attention output
    state_dict[f'layer{i}.mlp_fc1'] = matrix(2 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 2 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# ── primitive ops ──
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    m = max(v.data for v in logits)
    exps = [(v - m).exp() for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# ── GPT forward: one token at a time, with explicit KV cache ──
def gpt(token_id, pos_id, keys, values):
    # embeddings
    tok = state_dict['wte'][token_id]
    pos = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok, pos)]

    for li in range(n_layer):
        # ── multi-head causal attention ──
        res = x
        q = linear(rmsnorm(x), state_dict[f'layer{li}.attn_wq'])
        k = linear(rmsnorm(x), state_dict[f'layer{li}.attn_wk'])
        v = linear(rmsnorm(x), state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            qh = q[hs:hs+head_dim]
            kh = [ki[hs:hs+head_dim] for ki in keys[li]]
            vh = [vi[hs:hs+head_dim] for vi in values[li]]
            scores = [sum(qh[j] * kt[j] for j in range(head_dim)) / head_dim**0.5
                      for kt in kh]
            w = softmax(scores)
            out = [sum(w[t] * vh[t][j] for t in range(len(vh))) for j in range(head_dim)]
            x_attn.extend(out)
        x = [a + b for a, b in zip(linear(x_attn, state_dict[f'layer{li}.attn_wo']), res)]

        # ── feed-forward MLP ──
        res = x
        x = linear(rmsnorm(x), state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, res)]

    return linear(x, state_dict['lm_head'])   # logits over vocab

# ──────────────────────────────────────────────
# 5. Training Loop
# ──────────────────────────────────────────────
lr, beta1, beta2, eps = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)
num_steps = 500

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    losses = []

    for pos in range(n):
        logits = gpt(tokens[pos], pos, keys, values)
        probs = softmax(logits)
        losses.append(-probs[tokens[pos + 1]].log())   # cross-entropy

    loss = sum(losses) / n
    loss.backward()

    # Adam update
    lr_t = lr * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        mh = m[i] / (1 - beta1 ** (step + 1))
        vh = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * mh / (vh ** 0.5 + eps)
        p.grad = 0

    if step % 50 == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# ──────────────────────────────────────────────
# 6. Inference — autoregressive sampling
# ──────────────────────────────────────────────
temperature = 0.5
print("\n--- generated names ---")
for s in range(10):
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    token_id = BOS
    out = []
    for pos in range(block_size):
        logits = gpt(token_id, pos, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        if token_id < len(uchars):
            out.append(uchars[token_id])
    print(f"  {''.join(out)}")
