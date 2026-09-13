import os, time, zipfile, tempfile, shutil, urllib.request
from pathlib import Path

BASE = Path(r"C:\Users\shadb\Downloads\dataset")
ZIPS = BASE / "ipd37_zips"
OUTROOT = BASE / "IrishPotato37G"
RESULTS = BASE / "results"
for p in (ZIPS, OUTROOT, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

API = "https://zenodo.org/api/records/17553016/files/{key}/content"
FILES = [
    ("HEALTHY_1.zip", 778.1e6), ("HEALTHY_2.zip", 861.7e6), ("HEALTHY_3.zip", 830.0e6),
    ("HEALTHY_4.zip", 783.1e6), ("HEALTHY_5.zip", 690.1e6), ("HEALTHY_6.zip", 489.4e6),
    ("EARLYBLT_1.zip", 1950.7e6), ("EARLYBLT_2.zip", 2247.6e6), ("EARLYBLT_3.zip", 2343.7e6),
    ("EARLYBLT_4.zip", 2266.0e6), ("EARLYBLT_5.zip", 2245.0e6), ("EARLYBLT_6.zip", 2070.1e6),
    ("EARLYBLT_7.zip", 2316.6e6), ("EARLYBLT_8.zip", 3073.1e6), ("EARLYBLT_9.zip", 1756.4e6),
    ("LATEBLT_1.zip", 2025.2e6), ("LATEBLT_2.zip", 2109.9e6), ("LATEBLT_3.zip", 2288.4e6),
    ("LATEBLT_4.zip", 2408.3e6), ("LATEBLT_5.zip", 2096.1e6), ("LATEBLT_6.zip", 132.2e6),
]
IMG_EXT = (".jpg", ".jpeg", ".png")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(key, approx_size):
    dest = ZIPS / key
    marker = dest.with_suffix(dest.suffix + ".done")
    if marker.exists():
        return
    url = API.format(key=key)
    attempt = 0
    while True:
        attempt += 1
        have = dest.stat().st_size if dest.exists() else 0
        if have >= approx_size * 0.999:
            marker.touch()
            log(f"{key} complete ({have/1e6:.0f} MB)")
            return
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            got = dest.stat().st_size - have
            log(f"{key} +{got/1e6:.0f}MB in {time.time()-t0:.0f}s (total {dest.stat().st_size/1e6:.0f} MB)")
        except Exception as e:
            log(f"{key} retry {attempt}: {e}")
            time.sleep(min(60, 5 * attempt))


def class_of(zipname):
    n = zipname.upper()
    if "EARLYBLT" in n:
        return "earlyblt"
    if "LATEBLT" in n:
        return "lateblt"
    return "healthy"


def extract_and_prune():
    for key, _ in FILES:
        dest = ZIPS / key
        marker = dest.with_suffix(dest.suffix + ".done")
        extmark = dest.with_suffix(dest.suffix + ".extracted")
        if not marker.exists() or extmark.exists() or not dest.exists():
            continue
        cls = class_of(key)
        cdir = OUTROOT / cls
        cdir.mkdir(exist_ok=True)
        tag = Path(key).stem
        n = 0
        tmp = Path(tempfile.mkdtemp(dir=str(BASE)))
        try:
            with zipfile.ZipFile(dest) as z:
                z.extractall(tmp)
            for f in Path(tmp).rglob("*"):
                if f.is_file() and f.suffix.lower() in IMG_EXT:
                    target = cdir / f.name
                    if target.exists():
                        target = cdir / f"{tag}_{f.name}"
                        k = 1
                        while target.exists():
                            target = cdir / f"{tag}_{k}_{f.name}"
                            k += 1
                    shutil.move(str(f), str(target))
                    n += 1
            extmark.touch()
            dest.unlink()
            log(f"extracted {key}: {n} images -> {cls} (zip removed)")
        except Exception as e:
            log(f"EXTRACT FAIL {key}: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    for key, size in FILES:
        fetch(key, size)
        extract_and_prune()
    # final sweep
    extract_and_prune()
    counts = {}
    for cls in ("earlyblt", "healthy", "lateblt"):
        d = OUTROOT / cls
        counts[cls] = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT) if d.exists() else 0
    log(f"DONE class counts: {counts} | total {sum(counts.values())}")
    (RESULTS / "EXTRACT_DONE.flag").touch()


if __name__ == "__main__":
    main()
