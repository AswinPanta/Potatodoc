"""
Data preparation pipeline — scan, quarantine, dedup, split.
Reads dataset config from config/datasets.yaml.

Usage:
    python scripts/prepare_data.py --dataset irish
    python scripts/prepare_data.py --dataset plantvillage

Pipeline stages:
    1. Scan       — list all images per class
    2. Quarantine — PIL verify, move corrupt files to _quarantine/
    3. Dedup      — SHA256 exact duplicates (report only)
    4. Near-dedup — pHash grouping (Hamming <= 5), Union-Find
    5. Split      — group-aware stratified split → split_cache JSON
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import yaml
except ImportError:
    print("Missing pyyaml. Run: pip install pyyaml")
    sys.exit(1)

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("WARNING: imagehash not installed, skipping near-dedup. Run: pip install imagehash")

BASE_DIR = Path(__file__).resolve().parent.parent  # where this Python code is
CONFIG_PATH = BASE_DIR / "config" / "datasets.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NEAR_DUP_HAMMING = 5


# ============================================================
# File handling code: Read / create directory
# (independent of platform, created ahead, where Python code is)
# ============================================================
def ensure_dir(path: Path) -> Path:
    """Generate directory independent of platform, created ahead."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_path(p: str) -> Path:
    """Resolve config path relative to code location (platform-independent)."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (BASE_DIR / pp).resolve()


def load_dataset_cfg(dataset_id: str) -> dict:
    cfg = yaml.safe_load(open(CONFIG_PATH))
    if dataset_id not in cfg:
        raise ValueError(f"Unknown dataset '{dataset_id}'. Available: {list(cfg.keys())}")
    return cfg[dataset_id]


def stage_scan(root: Path, classes: list) -> pd.DataFrame:
    """Stage 1: scan all images per class directory."""
    records = []
    for cls_idx, cls_dir in enumerate(classes):
        cls_path = root / cls_dir
        if not cls_path.exists():
            print(f"  WARNING: {cls_path} missing, skipping")
            continue
        for f in sorted(cls_path.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                records.append({"path": str(f), "class_idx": cls_idx, "class": cls_dir})
    df = pd.DataFrame(records)
    print(f"  Scanned {len(df)} images")
    for cls_idx, cls_dir in enumerate(classes):
        n = (df["class_idx"] == cls_idx).sum()
        print(f"    {cls_dir}: {n}")
    return df


def stage_quarantine(df: pd.DataFrame, quarantine_dir: Path | None) -> pd.DataFrame:
    """Stage 2: PIL verify, move corrupt files to quarantine."""
    corrupt = []
    for _, row in df.iterrows():
        try:
            with Image.open(row["path"]) as img:
                img.verify()
        except Exception:
            corrupt.append(row["path"])

    print(f"  Corrupt images: {len(corrupt)}")
    if corrupt and quarantine_dir is not None:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for p in corrupt:
            src = Path(p)
            dst = quarantine_dir / src.name
            k = 1
            while dst.exists():
                dst = quarantine_dir / f"{src.stem}_{k}{src.suffix}"
                k += 1
            src.rename(dst)
        print(f"  Moved {len(corrupt)} files to {quarantine_dir}")
        df = df[~df["path"].isin(corrupt)].reset_index(drop=True)
    elif corrupt:
        df = df[~df["path"].isin(corrupt)].reset_index(drop=True)
        print("  Removed corrupt entries from dataframe (no quarantine dir)")
    return df


def stage_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 3: SHA256 exact duplicate detection (report only, keep first)."""
    print("  Computing SHA256 hashes...")
    t0 = time.time()
    hashes = []
    for p in df["path"]:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        hashes.append(h.hexdigest())
    df["sha256"] = hashes
    print(f"  Hashed {len(df)} images in {time.time()-t0:.1f}s")

    dup_mask = df["sha256"].duplicated(keep=False)
    n_dup_groups = df[dup_mask]["sha256"].nunique() if dup_mask.any() else 0
    print(f"  Exact duplicate images: {dup_mask.sum()} in {n_dup_groups} groups")

    if dup_mask.any():
        cross = 0
        for _, grp in df[dup_mask].groupby("sha256"):
            if grp["class_idx"].nunique() > 1:
                cross += 1
                print(f"  CROSS-CLASS DUP: {grp['class'].tolist()}")
        print(f"  Cross-class exact duplicates: {cross}")
    return df


def stage_near_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 4: pHash near-duplicate grouping via Union-Find."""
    if not HAS_IMAGEHASH:
        df["near_group"] = df["path"]
        return df

    print("  Computing perceptual hashes...")
    t0 = time.time()
    phashes = []
    for p in df["path"]:
        try:
            with Image.open(p) as img:
                phashes.append(str(imagehash.phash(img.convert("RGB"))))
        except Exception:
            phashes.append(None)
    df["phash"] = phashes
    print(f"  Done in {time.time()-t0:.1f}s")

    from imagehash import hex_to_hash
    valid = df[df["phash"].notna()].reset_index(drop=True)
    phash_objs = [hex_to_hash(h) for h in valid["phash"]]

    n = len(valid)
    print(f"  Checking {n} images for pHash distance <= {NEAR_DUP_HAMMING}...")
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_count = 0
    for i in range(n):
        hi = phash_objs[i]
        for j in range(i + 1, n):
            if hi - phash_objs[j] <= NEAR_DUP_HAMMING:
                union(valid.iloc[i]["path"], valid.iloc[j]["path"])
                pair_count += 1

    for p in df["path"]:
        find(p)
    df["near_group"] = df["path"].map(lambda x: find(x))
    group_sizes = df["near_group"].value_counts()
    print(f"  Near-duplicate pairs: {pair_count}")
    print(f"  Groups with >1 image: {(group_sizes > 1).sum()}, max size: {group_sizes.max()}")
    return df


def stage_split(df: pd.DataFrame, split: list, seed: int, cache_path: Path) -> dict:
    """Stage 5: group-aware stratified split, save paths to JSON cache."""
    group_rows = []
    for gid, g in df.groupby("near_group"):
        counts = g["class_idx"].value_counts()
        group_rows.append({
            "group": gid,
            "label": int(counts.index[0]),
            "n": len(g),
            "mixed": g["class_idx"].nunique() > 1,
        })
    groups_df = pd.DataFrame(group_rows)

    mixed = groups_df[groups_df["mixed"]]
    if len(mixed):
        print(f"  WARNING: {len(mixed)} near-duplicate groups contain multiple labels")

    train_ratio, val_ratio, test_ratio = split
    g_train, g_temp = train_test_split(
        groups_df["group"].tolist(), test_size=(1 - train_ratio),
        random_state=seed, stratify=groups_df["label"].tolist(),
    )
    gtemp_labels = groups_df.set_index("group").loc[g_temp, "label"].tolist()
    g_val, g_test = train_test_split(
        g_temp, test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=seed, stratify=gtemp_labels,
    )

    split_map = {g: "train" for g in g_train}
    split_map.update({g: "val" for g in g_val})
    split_map.update({g: "test" for g in g_test})
    df["split"] = df["near_group"].map(split_map)

    # Verify no leakage
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ga = set(df.loc[df["split"] == a, "near_group"])
        gb = set(df.loc[df["split"] == b, "near_group"])
        assert not (ga & gb), f"LEAKAGE: groups overlap between {a} and {b}"

    out = {
        "X_train": df.loc[df["split"] == "train", "path"].tolist(),
        "y_train": df.loc[df["split"] == "train", "class_idx"].tolist(),
        "X_val": df.loc[df["split"] == "val", "path"].tolist(),
        "y_val": df.loc[df["split"] == "val", "class_idx"].tolist(),
        "X_test": df.loc[df["split"] == "test", "path"].tolist(),
        "y_test": df.loc[df["split"] == "test", "class_idx"].tolist(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cache_path, "w"))
    print(f"  Split: train={len(out['X_train'])} val={len(out['X_val'])} test={len(out['X_test'])}")
    print(f"  Saved to {cache_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset id from config/datasets.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-near-dedup", action="store_true")
    args = ap.parse_args()

    cfg = load_dataset_cfg(args.dataset)
    root = resolve_repo_path(cfg["root"])  # platform-independent, relative to code
    print(f"Dataset: {args.dataset} root={root}")
    if not root.exists():
        print(f"ERROR: dataset root does not exist: {root}")
        sys.exit(1)

    print("\n[1/5] Scan")
    df = stage_scan(root, cfg["classes"])

    print("\n[2/5] Quarantine")
    qdir = (root / cfg["quarantine_dir"]) if cfg.get("quarantine_dir") else None
    df = stage_quarantine(df, qdir)

    print("\n[3/5] Exact dedup")
    df = stage_dedup(df)

    print("\n[4/5] Near dedup")
    if args.skip_near_dedup:
        df["near_group"] = df["path"]
        print("  Skipped (--skip-near-dedup)")
    else:
        df = stage_near_dedup(df)

    print("\n[5/5] Split")
    # Result -> always export to file (split cache); create directory ahead
    cache_path = resolve_repo_path(cfg["split_cache"])
    ensure_dir(cache_path.parent)
    stage_split(df, cfg["split"], args.seed, cache_path)
    print("\nDONE")


if __name__ == "__main__":
    main()
