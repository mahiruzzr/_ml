import sys
import math
import random
from collections import Counter

random.seed(42)

EOS = "<EOS>"
PAD = "<PAD>"

def load_corpus(filepath):
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    chars = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            chars.extend(list(line))
            chars.append(EOS)
    return chars

def build_ngram_counts(chars, n):
    counts = Counter()
    context_counts = Counter()
    for i in range(len(chars) - n):
        context = tuple(chars[i:i+n])
        next_char = chars[i+n]
        counts[(context, next_char)] += 1
        context_counts[context] += 1
    return counts, context_counts

class NgramLM:
    def __init__(self, n, smoothing=0.01):
        self.n = n
        self.smoothing = smoothing
        self.counts = None
        self.context_counts = None
        self.vocab = set()

    def train(self, chars):
        self.counts, self.context_counts = build_ngram_counts(chars, self.n)
        self.vocab = set(c for c in chars if not c.startswith("<"))

    def proba(self, context_chars):
        context = tuple(context_chars)
        total = self.context_counts.get(context, 0)
        probs = {}
        for c in self.vocab | {EOS}:
            c_count = self.counts.get((context, c), 0)
            probs[c] = (c_count + self.smoothing) / (total + self.smoothing * (len(self.vocab) + 1))
        return probs

def sample_next(context_chars, model, temperature=1.0, top_k=0):
    context = context_chars
    if len(context) < model.n:
        context = [PAD] * (model.n - len(context)) + list(context)
    else:
        context = list(context[-model.n:])

    probs = model.proba(tuple(context))

    chars = list(probs.keys())
    p = list(probs.values())

    if top_k and top_k < len(chars):
        top_indices = sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:top_k]
        chars = [chars[i] for i in top_indices]
        p = [p[i] for i in top_indices]

    if temperature != 1.0:
        log_p = [math.log(max(v, 1e-10)) / temperature for v in p]
        max_log = max(log_p)
        exp_p = [math.exp(v - max_log) for v in log_p]
        total = sum(exp_p)
        p = [v / total for v in exp_p]

    chosen = random.choices(chars, weights=p, k=1)[0]
    return chosen

def generate(prompt, model, max_len=30, temperature=1.0, top_k=5):
    generated = list(prompt)
    context = list(prompt)
    for _ in range(max_len):
        char = sample_next(context, model, temperature, top_k)
        if char == EOS:
            break
        generated.append(char)
        context.append(char)
        if len(context) > model.n:
            context = context[-model.n:]
    return "".join(generated)

def interactive(model):
    print("=" * 50)
    print(f"N-gram Language Model (n={model.n})")
    print("=" * 50)
    demos = ["小貓", "天上", "今天", "春天", "我"]
    for p in demos:
        print(f"  [{p}] → {generate(p, model, max_len=20, temperature=0.8)}")
    while True:
        try:
            raw = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!"); break
        if not raw or raw.lower() == "q":
            break
        print(f"  {generate(raw, model)}")

if __name__ == "__main__":
    corpus_file = sys.argv[1] if len(sys.argv) > 1 else "tw.txt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    chars = load_corpus(corpus_file)
    vocab = Counter(c for c in chars if not c.startswith("<"))
    print(f"corpus: {corpus_file}, chars: {len(chars)}, vocab: {len(vocab)}")
    model = NgramLM(n=n, smoothing=0.01)
    model.train(chars)
    print(f"trained ngram model (n={n})")
    interactive(model)
