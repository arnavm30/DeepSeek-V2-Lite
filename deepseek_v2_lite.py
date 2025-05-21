import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_rotary_embedding(seq_len, rotary_dim, device):
    """
    Generate sinusoidal position embeddings for rotary position encoding.
    Returns sine and cosine components needed for RoPE (Rotary Position Embeddings).
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim).float() / rotary_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, rotary_dim)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, 2 * rotary_dim)
    return torch.sin(emb)[None, None, :, :], torch.cos(emb)[None, None, :, :]

def apply_rotary_emb_single(x, sin, cos):
    """
    Applies Rotary Position Embeddings to a single tensor (query or key).
    
    RoPE works by rotating vector representations in the complex plane, 
    which allows the model to be aware of relative positions without
    adding additional parameters.
    
    Args:
        x: Input tensor to apply rotary embeddings to
        sin: Sine component of positional embedding
        cos: Cosine component of positional embedding
        
    Returns:
        Tensor with rotary embeddings applied
    """
    # Get rotary dimension from input tensor
    rot_dim = x.shape[-1] // 2
    
    # Make sure sin and cos match the rotary dimension
    sin = sin[..., :rot_dim]
    cos = cos[..., :rot_dim]
    
    # Split the vector into even and odd indices
    x1, x2 = x[..., ::2], x[..., 1::2]  # Even and odd dimensions
    
    # Perform the rotation in the complex plane
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Unlike LayerNorm, RMSNorm doesn't normalize using the mean,
    making it more computationally efficient.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x * self.weight / (norm / math.sqrt(x.shape[-1]) + self.eps)

class SwiGLU(nn.Module):
    """
    SwiGLU activation function (Swish-Gated Linear Unit).
    
    This is a key component used in modern LLMs like DeepSeek-V2.
    It combines the Swish activation (SiLU) with a gating mechanism.
    
    In the full DeepSeekMoE architecture, this activation would be used
    within each expert of the Mixture-of-Experts layers.
    
    Note: This simplified implementation doesn't include the full MoE routing,
    which would select different experts for different tokens.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # Project input to twice the hidden dimension to get both 
        # the projection and gate components
        self.proj = nn.Linear(dim, hidden_dim * 2)

    def forward(self, x):
        # Split the projected tensor into two parts along the last dimension
        x_proj, x_gate = self.proj(x).chunk(2, dim=-1)
        # Apply SiLU (Swish) activation to the gate and multiply by projection
        # SiLU(x) = x * sigmoid(x)
        return F.silu(x_gate) * x_proj

class FeedForward(nn.Module):
    """
    Feed-forward network with SwiGLU activation.
    
    In the full DeepSeek-V2 model, this would be replaced with a MoE layer
    where different tokens are routed to different experts.
    Each expert would have its own SwiGLU activation.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.swiglu = SwiGLU(dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.out_proj(self.swiglu(x))

class MLAAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA).
    
    MLA uses learned latent vectors instead of the traditional key-value attention.
    This significantly reduces memory requirements during inference by eliminating
    the need to store KV cache for all sequence positions.
    """
    def __init__(self, dim, n_heads, n_latents=64, rope_ratio=0.5):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_latents = n_latents
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # RoPE configuration - split head dimension into RoPE and non-RoPE parts
        self.rope_ratio = rope_ratio
        self.rope_dim = int(self.head_dim * rope_ratio)  # Dimension with RoPE applied
        self.non_rope_dim = self.head_dim - self.rope_dim  # Dimension without RoPE
        
        # Learned latent vectors
        # These replace the need to compute and store keys and values for every token
        self.latents = nn.Parameter(torch.randn(1, n_latents, dim))
        
        # Projection matrices
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, sin, cos, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            sin, cos: Rotary position embedding tensors
            mask: Ignored in MLA since we attend to learned latent vectors, not other tokens
        
        Returns:
            output: Attention output of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        
        # 1. Project inputs to queries and latents to key-values
        # ------------------------------------------------------
        q = self.q_proj(x)  # [batch_size, seq_len, dim]
        
        # Expand latent vectors to batch size
        latents = self.latents.expand(batch_size, -1, -1)  # [batch_size, n_latents, dim]
        kv = self.kv_proj(latents)  # [batch_size, n_latents, dim*2]
        
        # 2. Reshape for multi-head attention
        # -----------------------------------
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [batch_size, n_heads, seq_len, head_dim]
        
        kv = kv.view(batch_size, self.n_latents, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, batch_size, n_heads, n_latents, head_dim]
        k, v = kv[0], kv[1]  # Each: [batch_size, n_heads, n_latents, head_dim]
        
        # 3. Apply Rotary Position Embeddings
        # ------------------------------------------
        # For queries (seq_len dimension)
        sin_pos = sin[:, :, :seq_len, :]
        cos_pos = cos[:, :, :seq_len, :]
        
        # For keys (n_latents dimension)
        sin_latent = sin[:, :, :self.n_latents, :]
        cos_latent = cos[:, :, :self.n_latents, :]
        
        if self.rope_dim > 0:
            # Split query and key into RoPE and non-RoPE parts along head dimension
            # This is the "decoupled" RoPE approach from DeepSeek-V2 
            # which applies positional encoding to only part of the dimensions
            q_rope = q[..., :self.rope_dim]  # Part that will have positional encoding
            q_non_rope = q[..., self.rope_dim:]  # Part that will remain position-agnostic
            
            k_rope = k[..., :self.rope_dim]
            k_non_rope = k[..., self.rope_dim:]
            
            # Apply RoPE to queries with sequence position embeddings
            q_rope = apply_rotary_emb_single(q_rope, sin_pos, cos_pos)
            # Apply RoPE to keys with latent position embeddings
            k_rope = apply_rotary_emb_single(k_rope, sin_latent, cos_latent)
            
            # Recombine the RoPE and non-RoPE parts
            q = torch.cat([q_rope, q_non_rope], dim=-1)
            k = torch.cat([k_rope, k_non_rope], dim=-1)
        
        # 4. Compute attention
        # -------------------
        # Compute attention scores: [batch_size, n_heads, seq_len, n_latents]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Compute weighted sum of values: [batch_size, n_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # 5. Combine heads and project output
        # ----------------------------------
        # Reshape back to [batch_size, seq_len, dim]
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, dim)
        
        # Final projection
        output = self.out_proj(context)
        
        return output

class DecoderBlock(nn.Module):
    """
    Standard transformer decoder block with MLAAttention and FeedForward networks.
    In the full DeepSeek-V2 model, the FeedForward would be a MoE layer.
    """
    def __init__(self, dim, n_heads, ff_dim, n_latents=64):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = MLAAttention(dim, n_heads, n_latents)
        self.ln2 = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_dim)

    def forward(self, x, sin, cos, mask=None):
        """
        Forward pass through the decoder block.
        
        Args:
            x: Input tensor
            sin, cos: Positional embedding tensors
            mask: Attention mask (unused in MLA, kept for API compatibility)
            
        Returns:
            Output tensor after attention and feed-forward processing
        """
        # Apply attention with residual connection
        x = x + self.attn(self.ln1(x), sin, cos, mask)
        # Apply feed-forward with residual connection
        x = x + self.ff(self.ln2(x))
        return x

class DeepSeekV2Lite(nn.Module):
    """
    Simplified implementation of DeepSeek-V2 model.
    
    This implementation includes MLA but doesn't implement the full MoE architecture
    with expert routing. In the full DeepSeek-V2, FeedForward networks would be
    replaced with MoE layers that select different experts for different tokens.
    """
    def __init__(self, vocab_size=32000, dim=512, n_layers=6, n_heads=8, ff_dim=2048, 
                 max_seq_len=128, n_latents=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            DecoderBlock(dim, n_heads, ff_dim, n_latents) for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        self.max_seq_len = max_seq_len
        self.dim = dim
        self.n_heads = n_heads

    def forward(self, x):
        """
        Forward pass through the DeepSeekV2Lite model.
        
        Args:
            x: Input tensor of token IDs with shape (batch_size, seq_len)
            
        Returns:
            logits: Output tensor with shape (batch_size, seq_len, vocab_size)
        """
        B, T = x.shape
        x = self.token_emb(x)

        # Calculate rotary embeddings
        head_dim = self.dim // self.n_heads
        rotary_dim = head_dim // 2  # Half the head dimension for RoPE
        sin, cos = get_rotary_embedding(self.max_seq_len, rotary_dim, x.device)

        # MLA doesn't need causal mask since it attends to learned latent vectors
        mask = None

        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, sin, cos, mask)

        # Final normalization and projection to vocabulary
        logits = self.head(self.ln_f(x))
        return logits

if __name__ == "__main__":
    # Test the MLA model
    model = DeepSeekV2Lite(n_latents=64)
    
    x = torch.randint(0, 32000, (2, 128))  # (batch, seq_len)
    
    out = model(x)
    
    print(f"MLA model output shape: {out.shape}")  # Should be (2, 128, 32000)