"""Router head used by explicit-moe: self-attention over the MLLM instruction features, mean-pooled,
then a linear classifier over the edit categories (style, add, remove, replace, background)."""
import torch.nn as nn


class AttnHead(nn.Module):
    def __init__(self, d_in=2048, d=512, heads=8, dropout=0.1, ncls=5):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.ln0 = nn.LayerNorm(d)
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.ln3 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, ncls)

    def forward(self, x):                       # x [B, L, d_in]
        x = self.ln0(self.proj(x))
        a = self.ln1(x)
        a, _ = self.attn(a, a, a, need_weights=False)   # bidirectional, no mask (batches have no padding)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return self.fc(self.ln3(x.mean(1)))
