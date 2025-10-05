import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def causal_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


class TokenEmbedding:
    def __init__(self, vocab_size, d_model, rng=np.random):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.W = rng.normal(scale=0.02, size=(vocab_size, d_model))

    def __call__(self, token_ids):
        return self.W[token_ids]


class SinusoidalPositionalEncoding:
    def __init__(self, d_model, max_len=512):
        self.d_model = d_model
        self.max_len = max_len
        self.PE = self._build_pe(max_len, d_model)

    def _build_pe(self, max_len, d_model):
        pe = np.zeros((max_len, d_model))
        positions = np.arange(max_len).reshape(-1, 1)
        even_i = np.arange(0, d_model, 2)
        denominator = np.power(10000.0, (2 * even_i) / d_model)
        angle_rads = positions / denominator
        pe[:, 0::2] = np.sin(angle_rads)
        pe[:, 1::2] = np.cos(angle_rads)
        return pe
    
    def __call__(self, seq_len):
        if seq_len > self.max_len:
            raise ValueError("seq_len > max_len in positional encoding")
        return self.PE[:seq_len][np.newaxis, :, :]


class ScaledDotProductAttention:
    def __init__(self):
        pass
    
    def __call__(self, Q, K, V, mask=None):
        dk = Q.shape[-1]
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(dk)
        if mask is not None:
            mask_expand = mask[None, None, :, :]
            scores = np.where(mask_expand, scores, -1e9)
        attn = softmax(scores, axis=-1)
        out = np.matmul(attn, V)
        return out, attn


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, rng=np.random):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.Wq = rng.normal(scale=0.02, size=(d_model, d_model))
        self.Wk = rng.normal(scale=0.02, size=(d_model, d_model))
        self.Wv = rng.normal(scale=0.02, size=(d_model, d_model))
        self.Wo = rng.normal(scale=0.02, size=(d_model, d_model))

        self.attn_module = ScaledDotProductAttention()

    def split_heads(self, x):
        b, s, _ = x.shape
        x = x.reshape(b, s, self.num_heads, self.dk)
        return np.transpose(x, (0,2,1,3))
    
    def combine_heads(self, x):
        x = np.transpose(x, (0,2,1,3))
        b, s, h, dk = x.shape
        return x.reshape(b, s, h*dk)
    
    def __call__(self, x, mask=None, return_attn=False):
        Q = np.matmul(x, self.Wq)
        K = np.matmul(x, self.Wk)
        V = np.matmul(x, self.Wv)

        Qh = self.split_heads(Q)
        Kh = self.split_heads(K)
        Vh = self.split_heads(V)

        out, attn = self.attn_module(Qh, Kh, Vh, mask=mask)
        concat = self.combine_heads(out)
        projected = np.matmul(concat, self.Wo)
        return (projected, attn) if return_attn else projected


class FeedForward:
    def __init__(self, d_model, d_ff, rng=np.random):
        self.W1 = rng.normal(scale=0.02, size=(d_model, d_ff))
        self.b1 = np.zeros((d_ff,))
        self.W2 = rng.normal(scale=0.02, size=(d_ff, d_model))
        self.b2 = np.zeros((d_model,))

    def __call__(self, x):
        h = np.matmul(x, self.W1) + self.b1
        h = np.maximum(0, h)
        out = np.matmul(h, self.W2) + self.b2
        return out


class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones((d_model,))
        self.beta = np.zeros((d_model,))

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff, rng=np.random):
        self.mha = MultiHeadAttention(d_model, num_heads, rng=rng)
        self.ln1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, rng=rng)
        self.ln2 = LayerNorm(d_model)
        
    def __call__(self, x, mask=None, return_attn=False):
        x_norm = self.ln1(x)
        attn_out, attn_weights = self.mha(x_norm, mask=mask, return_attn=True)
        x = x + attn_out
        x_norm2 = self.ln2(x)
        ff_out = self.ff(x_norm2)
        x = x + ff_out
        if return_attn:
            return x, attn_weights
        return x


class DecoderOnlyTransformer:
    def __init__(self, vocab_size, d_model=256, num_heads=4, d_ff=256, num_layers=2, max_len=512, rng=np.random, weight_tie=False):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_emb = TokenEmbedding(vocab_size, d_model, rng=rng)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.layers = [DecoderBlock(d_model, num_heads, d_ff, rng=rng) for _ in range(num_layers)]
        self.ln_f = LayerNorm(d_model)

        if weight_tie:
            self.W_out = self.token_emb.W.T  
        else:
            self.W_out = rng.normal(scale=0.02, size=(d_model, vocab_size))

    def forward(self, token_ids, return_attn=False):
        b, s = token_ids.shape
        x = self.token_emb(token_ids)
        x = x + self.pos_enc(s)
        attn_weights_all = []
        mask = causal_mask(s)

        for layer in self.layers:
            if return_attn:
                x, attn = layer(x, mask=mask, return_attn=True)
                attn_weights_all.append(attn)
            else:
                x = layer(x, mask=mask, return_attn=False)

        x = self.ln_f(x)
        logits = np.matmul(x, self.W_out)
        last_token_logits = logits[:, -1, :]
        next_token_probs = softmax(last_token_logits, axis=-1)
        if return_attn:
            return logits, next_token_probs, attn_weights_all
        return logits, next_token_probs


if __name__ == "__main__":
    np.random.seed(42)
    B = 4              # batch size
    S = 20             # sequence length
    VOCAB = 1000       # vocabulary size
    D = 512            # embedding dimension
    HEADS = 8          # number of attention heads
    D_FF = 2048        # hidden dimension in feed-forward
    LAYERS = 6         # number of decoder layers

    print("=== Transformer Decoder-Only Model Test ===\n")

    print("Hyperparameters:")
    print(f"  vocab_size={VOCAB}")
    print(f"  d_model={D}")
    print(f"  num_heads={HEADS}")
    print(f"  d_ff={D_FF}")
    print(f"  num_layers={LAYERS}")
    print(f"  max_seq_len={S}\n")

    dummy_input = np.random.randint(0, VOCAB, size=(B, S))
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB,
        d_model=D,
        num_heads=HEADS,
        d_ff=D_FF,
        num_layers=LAYERS,
        max_len=128
    )

    print(f"Input:{dummy_input}\n")

    # forward pass
    logits, next_token_probs, attn_weights = model.forward(dummy_input, return_attn=True)

    # hasil akhir
    print("Output logits shape:", logits.shape, "(batch_size, seq_len, vocab_size)")
    print("Next token probabilities shape:", next_token_probs.shape, "(batch_size, vocab_size)")
    print("Sum of probabilities for each batch:", np.sum(next_token_probs, axis=-1))

    # validasi softmax
    assert np.allclose(np.sum(next_token_probs, axis=-1), 1, atol=1e-6)
    print("\n✅ Forward pass successful and softmax validation passed.")