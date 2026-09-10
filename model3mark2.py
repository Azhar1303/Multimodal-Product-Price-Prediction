# Required libraries
import os, torch, random, numpy as np, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import open_clip
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

# Input file directories
BASE_DIR = "/DATA1/ai24mtech12010/elevenOct"
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
OUT_DIR = os.path.join(BASE_DIR, "vlm_runs_v4")
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = 12
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MODEL_NAME = "ViT-L-14"
PRETRAINED = "datacomp_xl_s13b_b90k"  
UNFREEZE_RATIO = 0.5

# This is to ensure reproducibility of results
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#  Dataset optimization
class ProductDataset(Dataset):
    def __init__(self, df, img_dir, preprocess, tokenizer, max_len=77):
        self.df, self.img_dir, self.preprocess, self.tokenizer = df, img_dir, preprocess, tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row["sample_id"])
        txt = str(row["catalog_content"]) if pd.notna(row["catalog_content"]) else ""
        price = torch.tensor(row.get("price", 0.0), dtype=torch.float32)
        path = os.path.join(self.img_dir, f"{sid}.jpg")
        if not os.path.exists(path):
            path = os.path.join(self.img_dir, f"{sid}.png")
        if not os.path.exists(path):
            img = Image.new("RGB", (224,224), (0,0,0))
        else:
            img = Image.open(path).convert("RGB")
        return self.preprocess(img), txt, price, sid

# Encoding long text inputs for the vision–language model (VLM)
def encode_long_text(vlm, tokenizer, texts, max_len=77, device="cuda"):
    toks = tokenizer(texts).to(device)  # Accessing GPU
    if toks.shape[1] <= max_len:
        return vlm.encode_text(toks)
    chunks = []
    for start in range(0, toks.shape[1], max_len - 10):
        chunks.append(vlm.encode_text(toks[:, start:start + max_len].to(device)))  
    return torch.stack(chunks, dim=0).mean(0)


# Neural network head that combines image and text embeddings to predict product price
class PriceHead(nn.Module):
    def __init__(self, in_dim_img, in_dim_txt, hidden=512, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim_img + in_dim_txt, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Softplus()
        )
    # Normalize embeddings, concatenate them, and output predicted price
    def forward(self, img_emb, txt_emb):
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-12)
        txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-12)
        return self.fc(torch.cat([img_emb, txt_emb], dim=-1))

#  Main training function 
def train():
    print(f"Loading VLM: {MODEL_NAME} | {PRETRAINED}")
    vlm, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    # Freeze first part of layers(image)
    total_blocks = len(vlm.visual.transformer.resblocks)
    freeze_upto = int(total_blocks * (1 - UNFREEZE_RATIO))
    for i, blk in enumerate(vlm.visual.transformer.resblocks):
        for p in blk.parameters():
            p.requires_grad = i >= freeze_upto
    # Same for text encoder
    total_tblocks = len(vlm.transformer.resblocks)
    freeze_upto_txt = int(total_tblocks * (1 - UNFREEZE_RATIO))
    for i, blk in enumerate(vlm.transformer.resblocks):
        for p in blk.parameters():
            p.requires_grad = i >= freeze_upto_txt

    head = PriceHead(vlm.visual.output_dim, vlm.text_projection.shape[1]).to(DEVICE)

    # Dataset loading
    df = pd.read_csv(TRAIN_CSV)
    df = df[df["price"] > 0].reset_index(drop=True)
    val_split = int(0.9 * len(df))
    tr_df, val_df = df[:val_split], df[val_split:]
    tr_ds = ProductDataset(tr_df, TRAIN_IMG_DIR, preprocess, tokenizer)
    val_ds = ProductDataset(val_df, TRAIN_IMG_DIR, preprocess, tokenizer)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Optimizer + Scheduler
    optimizer = AdamW([
        {"params": [p for n,p in vlm.named_parameters() if p.requires_grad], "lr": 1e-5},
        {"params": head.parameters(), "lr": 1e-3}
    ])
    base_sched = CosineAnnealingLR(optimizer, T_max=EPOCHS - 1)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=1)
    scheduler = SequentialLR(optimizer, [warmup, base_sched], milestones=[1])

    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.SmoothL1Loss(beta=0.25) 

    best_rmse = 1e9
    for ep in range(1, EPOCHS + 1):
        vlm.train(); head.train()
        tr_losses = []
        for imgs, texts, prices, _ in tqdm(tr_dl, desc=f"Train Ep{ep:02d}"):
            imgs, prices = imgs.to(DEVICE), prices.to(DEVICE)
            with torch.amp.autocast('cuda'):
                img_emb = vlm.encode_image(imgs)
                txt_emb = encode_long_text(vlm, tokenizer, texts, device = DEVICE)
                pred_log = head(img_emb, txt_emb).squeeze(1)
                loss = criterion(pred_log, torch.log1p(prices))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            tr_losses.append(loss.item())
        scheduler.step()

        # Validation set exploration
        vlm.eval(); head.eval()
        val_preds, val_tgts = [], []
        with torch.no_grad():
            for imgs, texts, prices, _ in tqdm(val_dl, desc=f"Val Ep{ep:02d}"):
                imgs, prices = imgs.to(DEVICE), prices.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    img_emb = vlm.encode_image(imgs)
                    txt_emb = encode_long_text(vlm, tokenizer, texts)
                    pred = torch.expm1(head(img_emb, txt_emb).squeeze(1))
                val_preds.append(pred.cpu().numpy())
                val_tgts.append(prices.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_tgts = np.concatenate(val_tgts)
        rmse = np.sqrt(((val_preds - val_tgts) ** 2).mean())
        print(f"[{ep:02d}] TrainLoss={np.mean(tr_losses):.4f} | ValRMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({
                "vlm_state": vlm.state_dict(),
                "head_state": head.state_dict(),
                "cfg": {"model_name": MODEL_NAME, "pretrained": PRETRAINED}
            }, os.path.join(OUT_DIR, "best_vlm_v4.pth"))
            print(f"Saved best model | RMSE={best_rmse:.4f}")

    print(f"Training complete | Best Val RMSE={best_rmse:.4f}")

# Main function 
if __name__ == "__main__":
    train()
