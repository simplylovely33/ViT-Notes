'''
This manuscript references "Vision Transformers from Scratch (PyTorch): A step-by-step guide",
which is a guidance to train a ViT on MNIST dataset.
(https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)
'''
import copy
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
    """
    Confirm the device availability information
    """
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        print(f"[INFO]: Using {torch.cuda.get_device_name(device_id)} as default device.")
        device = torch.device('cuda')
    else:
        print(f"[INFO]: Using {torch.device('cpu')} as default device.")
        device = torch.device('cpu')
    return device

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def patchify(images, n_patches):
    n, c , h, w = images.shape
    assert h == w, "Patchify method is implemented for square images only"
    patches = torch.zeros(n, n_patches ** 2, c * h * w // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), : ].expand(x.size(0), -1, -1)
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=2, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0, f"Can't divide dimension {d_model} into {n_heads} heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q =  nn.Parameter(torch.randn(n_heads, self.d_k, d_model))
        self.W_k =  nn.Parameter(torch.randn(n_heads, self.d_k, d_model))
        self.W_v =  nn.Parameter(torch.randn(n_heads, self.d_k, d_model))
        self.W_o =  nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = torch.einsum("bnd,hkd->bh nk", x, self.W_q)
        key = torch.einsum("bnd,hkd->bh nk", x, self.W_k)
        value = torch.einsum("bnd,hkd->bh nk", x, self.W_v)

        # 2) Apply scaled dot-product attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) Concat heads → (B, N, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.n_heads * self.d_k)

        return self.W_o(x)

class SubLayerConnectionBlock(nn.Module):
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
        assert chw[1] % n_patches == 0, "Input image size must be divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input image size must be divisible by number of patches"
        
        self.patch_size = (self.chw[1] / n_patches, self.chw[2] / n_patches)

        # 1) Linear Mapper
        self.input_d = int(self.chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.positional_encoding = PositionalEncoding(self.hidden_d, 0)

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([SubLayerConnectionBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        # 5) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to('cuda')
        tokens = self.linear_mapper(patches)

        # Adding classification toekn to the tokens
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

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
        (1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10
    ).to(device)
    N_EPOCHS = 10
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
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

