# Required libraries
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import open_clip
from torch.utils.data import Dataset, DataLoader

# Input file directories
BASE = "/DATA1/ai24mtech12010/elevenOct"
DATA_DIR = os.path.join(BASE, "student_resource", "dataset")
OUT_DIR = os.path.join(BASE, "vlm_runs_v4")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
CKPT = os.path.join(OUT_DIR, "best_vlm_v4.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE}")

#  Dataset optimization
class ProductTest(Dataset):
    def __init__(self, df, img_dir, preprocess, tokenizer):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row["sample_id"])
        txt = str(row["catalog_content"]) if pd.notna(row["catalog_content"]) else ""
        path = os.path.join(self.img_dir, f"{sid}.jpg")
        if not os.path.exists(path):
            path = os.path.join(self.img_dir, f"{sid}.png")
        if not os.path.exists(path):
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            img = Image.open(path).convert("RGB")
        return self.preprocess(img), txt, sid

# Encoding long text inputs for the vision–language model (VLM)
def encode_long_text(vlm, tokenizer, texts, max_len=77, device="cuda"):
    toks = tokenizer(texts).to(device)
    if toks.shape[1] <= max_len:
        return vlm.encode_text(toks)
    chunks = []
    for start in range(0, toks.shape[1], max_len - 10):
        chunks.append(vlm.encode_text(toks[:, start:start + max_len].to(device)))
    return torch.stack(chunks, 0).mean(0)


# Neural network head that combines image and text embeddings to predict product price
class PriceHead(torch.nn.Module):
    def __init__(self, in_img, in_txt, hidden=512, dropout=0.1):
        super().__init__()
        self.fc = torch.nn.Sequential(        #  must be self.fc not self.net
            torch.nn.Linear(in_img + in_txt, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
            torch.nn.Softplus()
        )

    def forward(self, img_emb, txt_emb):
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-12)
        txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-12)
        return self.fc(torch.cat([img_emb, txt_emb], dim=-1))


# Loading Model and performing prediction
@torch.no_grad()
def main():
    te = pd.read_csv(TEST_CSV)
    ckpt = torch.load(CKPT, map_location=DEVICE)
    cfg = ckpt["cfg"]

    vlm, _, preprocess = open_clip.create_model_and_transforms(
        cfg["model_name"], pretrained=cfg["pretrained"], device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(cfg["model_name"])
    head = PriceHead(vlm.visual.output_dim, vlm.text_projection.shape[1]).to(DEVICE)

    vlm.load_state_dict(ckpt["vlm_state"])
    head.load_state_dict(ckpt["head_state"])
    vlm.eval(); head.eval()

    ds = ProductTest(te, TEST_IMG_DIR, preprocess, tokenizer)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    preds, ids = [], []
    for imgs, texts, sids in tqdm(dl, desc="Predicting"):
        imgs = imgs.to(DEVICE)
        txt_emb = encode_long_text(vlm, tokenizer, texts, device=DEVICE)
        with torch.amp.autocast('cuda'):
            img_emb = vlm.encode_image(imgs)
            pred_log = head(img_emb, txt_emb)
            prices = torch.expm1(pred_log)
            prices = torch.clamp(prices, min=1.0)
        preds.append(prices.squeeze(1).cpu().numpy())
        ids.extend(sids)

    preds = np.concatenate(preds)
    sub = pd.DataFrame({"sample_id": ids, "price": preds})
    out_csv = os.path.join(OUT_DIR, "submission_v4.csv")
    sub.to_csv(out_csv, index=False)
    print(f"Saved submission → {out_csv}")

if __name__ == "__main__":
    main()
