import math
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)

def device_confirmation():
    "Confirm the device availability information."
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        print(f"[INFO]: Using {torch.cuda.get_device_name(device_id)} as default device.")
        device = torch.device('cuda')
    else:
        print(f"[INFO]: Using {torch.device('cpu')} as default device.")
        device = torch.device('cpu')
    return device

def Patchify(images, n_patches):
    """
    images: (B, C, H, W)
    returns: (B, n_patches**2, patch_dim)
    """
    n, c, h, w = images.shape
    assert h == w, "Patchify method is implemented for square images only"
    assert h % n_patches == 0, "Image size must be divisible by n_patches"
    patch_size = h // n_patches
    patch_dim = c * patch_size * patch_size
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches shape: (B, C, n_patches, n_patches, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, n_patches, n_patches, C, p, p)
    patches = patches.view(n, n_patches * n_patches, patch_dim)
    return patches

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe[:x.size(1), :].unsqueeze(0)
        return x + pe

def attention(query, key, value, mask=None, dropout=None):
    "Compute Scaled Dot Product Attention.'"
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 3 and mask.size(1) == 1:
            mask = mask.unsqueeze(2)
    return nn.functional.scaled_dot_product_attention(
        query, key, value,
        attn_mask=mask,
        dropout_p=dropout.p if dropout is not None and dropout.training else 0.0,
        is_causal=False # If decoder autoregressive, set to True
    )

class MultiHeadSelfAttention(nn.Module):
    "Implement the Multi-Head Self-Attention mechanism."
    def __init__(self, d_model, n_heads=2, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0, f"Can't divide dimension {d_model} into {n_heads} heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q =  nn.Linear(d_model, d_model, bias=False)
        self.w_k =  nn.Linear(d_model, d_model, bias=False)
        self.w_v =  nn.Linear(d_model, d_model, bias=False)
        self.w_o =  nn.Linear(d_model, d_model, bias=False)

        self.attn = None
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, x, mask=None):
        B, N, _ = x.size()
        if mask is not None:
            mask = mask.unsqueeze(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        q = self.w_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x = attention(q, k, v, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.n_heads * self.d_k)

        return self.w_o(x)

class SubLayerBlock(nn.Module):
    "Implement the Sub-Layer Block Workflow."
    def __init__(self, hidden_d, n_heads, mlp_ratio=4 ):
        super().__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MultiHeadSelfAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out 

class VisionTransformer(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7, hidden_d=8, n_heads=2, n_blocks=2, out_d=10):
        # Super constructor
        super(VisionTransformer, self).__init__()
        
        # Attribute
        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.hidden_d = hidden_d
        assert chw[1] % n_patches == 0 and chw[2] % n_patches == 0, ""
        "Input image size must be divisible by number of patches"

        self.patch_size = (self.chw[1] / n_patches, self.chw[2] / n_patches)
        self.input_d = int(self.chw[0] * self.patch_size[0] * self.patch_size[1])

        # 1) Linear Mapper
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_d))
        
        # 3) Positional embedding
        seq_length = n_patches ** 2 + 1
        self.positional_encoding = PositionalEncoding(self.hidden_d, max_len=seq_length)

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([SubLayerBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        # 5) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

        # initialize parameters
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.class_token, std=0.02)
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'class_token' not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, images):
        B, C, H, W = images.shape
        device = images.device
        patches = Patchify(images, self.n_patches).to(device)
        tokens = self.linear_mapper(patches)

        cls = self.class_token.expand(B, -1, -1)
        # Adding classification toekn to the tokens
        tokens = torch.cat([cls, tokens], dim=1)

        # Adding positional embedding
        out = self.positional_encoding(tokens)

         # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out)
    
    
def main():
    # Loading data
    transform = ToTensor()
    train_set = MNIST(
        root='./../datasets', 
        train=True, download=True, transform=transform)
    test_set = MNIST(
        root='./../datasets', 
        train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = device_confirmation()
    model = VisionTransformer(
        (1, 28, 28), 
        n_patches=7,
        n_blocks=2, 
        hidden_d=8, 
        n_heads=2, 
        out_d=10
    ).to(device)

    N_EPOCHS = 5

    # Training loop
    #optimizer = Adam(model.parameters(), lr=LR)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=N_EPOCHS,
    )

    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")

if __name__ == "__main__":
    main()

