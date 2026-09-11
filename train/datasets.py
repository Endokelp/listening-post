"""Phase 3 dataset fetcher. Pulls three public datasets into data/raw/ (gitignored)
and writes normalized 48kHz/mono/16-bit copies into data/train/ and data/test/.

Run: python train/datasets.py
Self-check only: python train/datasets.py --self-check
"""
import csv
import json
import math
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
TARGET_SR = 48000

ALEMADI_URL = "https://github.com/saraalemadi/DroneAudioDataset"
SVANSTROM_URL = "https://github.com/DroneDetectionThesis/Drone-detection-dataset"
ESC50_URL = "https://github.com/karolpiczak/ESC-50"
DRONENOISE_API = "https://api.figshare.com/v2/articles/22133411"
DRONENOISE_MICS = ("M1", "M5", "M9")  # edge, centre, edge of the 9-mic array
UAV32_URL = "https://github.com/mackenzie-jane/drone-visualization"

ESC50_CONFUSER_CATS = {
    "engine", "chainsaw", "airplane", "helicopter", "hand_saw",
    "vacuum_cleaner", "washing_machine", "train", "insects",
}
ESC50_BACKGROUND_CATS = {
    "wind", "rain", "chirping_birds", "crickets", "sea_waves",
    "thunderstorm", "crackling_fire",
}


# --- naming -------------------------------------------------------------

def out_name(kind, source, label, recording):
    fields = {"kind": kind, "source": source, "label": label, "recording": recording}
    for name, val in fields.items():
        if not val:
            raise ValueError(f"field {name!r} is empty")
        if "__" in val:
            raise ValueError(f"field {name!r}={val!r} contains '__'")
    return f"{kind}__{source}__{label}__{recording}.wav"


def parse_name(filename):
    stem = filename[:-4] if filename.endswith(".wav") else filename
    parts = stem.split("__")
    if len(parts) != 4:
        raise ValueError(f"expected 4 fields (kind__source__label__recording), got {len(parts)}: {filename}")
    kind, source, label, recording = parts
    return {"kind": kind, "source": source, "label": label, "recording": recording}


def _self_check():
    name = out_name("drone", "alemadi", "bebop", "B_S2_D1_067")
    assert name == "drone__alemadi__bebop__B_S2_D1_067.wav", name
    assert parse_name(name) == {
        "kind": "drone", "source": "alemadi", "label": "bebop", "recording": "B_S2_D1_067",
    }
    try:
        out_name("drone", "alemadi", "be__bop", "x")
        assert False, "expected ValueError for '__' inside a field"
    except ValueError:
        pass
    try:
        out_name("drone", "alemadi", "", "x")
        assert False, "expected ValueError for empty field"
    except ValueError:
        pass
    try:
        parse_name("only__three__fields.wav")
        assert False, "expected ValueError for wrong field count"
    except ValueError:
        pass
    assert _uav32_label("DJI_Mini3_pro_30") == "dji-mini3-pro"
    assert _uav32_label("Syma_X8SW_87") == "syma-x8sw"
    assert _uav32_label("NoTrailingNumber") == "notrailingnumber"
    print("self-check passed")


# --- audio helpers --------------------------------------------------------

def downmix(audio):
    return audio.mean(axis=1) if audio.ndim > 1 else audio


def resample_to_48k(audio, sr):
    if sr == TARGET_SR:
        return audio
    g = math.gcd(sr, TARGET_SR)
    return resample_poly(audio, TARGET_SR // g, sr // g)


def write_wav(path, audio, source_sr):
    audio = resample_to_48k(downmix(audio), source_sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), TARGET_SR, subtype="PCM_16")


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd), check=True)


# --- dataset 1: Al-Emadi (drones -> train) --------------------------------

def fetch_alemadi():
    dest = RAW / "alemadi"
    if dest.exists():
        return dest
    git("clone", "--depth", "1", "--filter=blob:none", "--sparse", ALEMADI_URL, dest, cwd=ROOT)
    git("sparse-checkout", "set", "Multiclass_Drone_Audio/bebop_1", "Multiclass_Drone_Audio/membo_1", cwd=dest)
    return dest


def _alemadi_recordings(folder, raw_label):
    """Group original (non-augmented) segment files by source recording.
    Filenames look like '<recording>-<raw_label>_<seg>_.wav'; one oddball
    batch ('extra_membo_D2_2000.wav'...) has no '-label_' suffix, just a
    trailing running index. 'mixed_*' files are noise-augmented and excluded."""
    groups = defaultdict(list)
    for f in sorted(folder.glob("*.wav")):
        stem = f.stem
        if stem.startswith("mixed"):
            continue
        suffix = f"-{raw_label}_"
        if suffix in stem:
            recording, rest = stem.split(suffix, 1)
            seg = int(rest.rstrip("_"))
        else:
            recording, _, rest = stem.rpartition("_")
            seg = int(rest)
        groups[recording].append((seg, f))
    for recording in groups:
        groups[recording].sort()
    return groups


def process_alemadi():
    dest = fetch_alemadi()
    for folder_name, raw_label, out_label in [
        ("bebop_1", "bebop", "bebop"),
        ("membo_1", "membo", "mambo"),
    ]:
        recordings = _alemadi_recordings(dest / "Multiclass_Drone_Audio" / folder_name, raw_label)
        for recording, segs in recordings.items():
            out_path = TRAIN_DIR / out_name("drone", "alemadi", out_label, recording)
            if out_path.exists():
                continue
            chunks = []
            sr = None
            for _, f in segs:
                audio, sr = sf.read(str(f), dtype="float64")
                chunks.append(audio)
            write_wav(out_path, np.concatenate(chunks), sr)


# --- dataset 2: Svanström (drone/helicopter/background -> test) ----------

SVANSTROM_MAP = {
    "DRONE": ("drone", "drone"),
    "HELICOPTER": ("confuser", "helicopter"),
    "BACKGROUND": ("background", "outdoor"),
}


def fetch_svanstrom():
    dest = RAW / "svanstrom"
    if dest.exists():
        return dest
    git("clone", "--depth", "1", "--filter=blob:none", "--sparse", SVANSTROM_URL, dest, cwd=ROOT)
    git("sparse-checkout", "set", "Data/Audio", cwd=dest)
    return dest


def process_svanstrom():
    dest = fetch_svanstrom()
    for f in sorted((dest / "Data" / "Audio").glob("*.wav")):
        prefix = f.stem.split("_")[0]
        kind, label = SVANSTROM_MAP[prefix]
        out_path = TEST_DIR / out_name(kind, "svanstrom", label, f.stem)
        if out_path.exists():
            continue
        audio, sr = sf.read(str(f), dtype="float64")
        write_wav(out_path, audio, sr)


# --- dataset 3: ESC-50 (confusers + background; fold 5 -> test) ----------

def fetch_esc50():
    dest = RAW / "esc50"
    if dest.exists():
        return dest
    git("clone", "--depth", "1", "--filter=blob:none", "--sparse", ESC50_URL, dest, cwd=ROOT)
    git("sparse-checkout", "set", "--no-cone", "/meta/esc50.csv", cwd=dest)
    rows = _read_esc50_csv(dest)
    wanted = [r for r in rows if r["category"] in ESC50_CONFUSER_CATS or r["category"] in ESC50_BACKGROUND_CATS]
    patterns = "/meta/esc50.csv\n" + "\n".join(f"/audio/{r['filename']}" for r in wanted) + "\n"
    subprocess.run(["git", "sparse-checkout", "set", "--no-cone", "--stdin"],
                    cwd=str(dest), input=patterns, text=True, check=True)
    return dest


def _read_esc50_csv(dest):
    with (dest / "meta" / "esc50.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def process_esc50():
    dest = fetch_esc50()
    for row in _read_esc50_csv(dest):
        category = row["category"]
        if category in ESC50_CONFUSER_CATS:
            kind = "confuser"
        elif category in ESC50_BACKGROUND_CATS:
            kind = "background"
        else:
            continue
        split_dir = TEST_DIR if row["fold"] == "5" else TRAIN_DIR
        recording = row["filename"][:-4]
        out_path = split_dir / out_name(kind, "esc50", category, recording)
        if out_path.exists():
            continue
        src = dest / "audio" / row["filename"]
        audio, sr = sf.read(str(src), dtype="float64")
        write_wav(out_path, audio, sr)


# --- dataset 4: DroneNoise (outdoor ground mics -> train) ----------------

def _http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def fetch_dronenoise():
    dest = RAW / "dronenoise"
    if dest.exists():
        return dest
    dest.mkdir(parents=True)
    files = json.loads(_http_get(DRONENOISE_API))["files"]
    mic_suffixes = tuple(f"_{m}.wav" for m in DRONENOISE_MICS)
    for f in files:
        name = f["name"]
        if name.startswith("Calib_"):
            continue
        if name.endswith(".xlsx") or name.endswith(mic_suffixes):
            (dest / name).write_bytes(_http_get(f["download_url"]))
    return dest


def process_dronenoise():
    """Filenames are <event>_M<n>.wav, one per mic recording the same event.
    Group them under a shared 'event@mic' recording id (see out_name below;
    windows.py strips '@mic' when forming cross-validation groups)."""
    dest = fetch_dronenoise()
    for f in sorted(dest.glob("*.wav")):
        event, _, mic = f.stem.rpartition("_")
        label = f.stem.split("_")[1].lower()
        out_path = TRAIN_DIR / out_name("drone", "dronenoise", label, f"{event}@{mic}")
        if out_path.exists():
            continue
        audio, sr = sf.read(str(f), dtype="float64")
        write_wav(out_path, audio, sr)


# --- dataset 5: 32-airframe set (cross-airframe -> test) -----------------

def fetch_uav32():
    dest = RAW / "uav32"
    if dest.exists():
        return dest
    git("clone", "--depth", "1", "--filter=blob:none", "--sparse", UAV32_URL, dest, cwd=ROOT)
    git("sparse-checkout", "set", "public/droneAudio", cwd=dest)
    return dest


def _uav32_label(stem):
    """'DJI_Mini3_pro_30' -> 'dji-mini3-pro' (drop trailing _<number>, '_'->'-')."""
    base, _, tail = stem.rpartition("_")
    base = base if tail.isdigit() else stem
    return base.lower().replace("_", "-")


def process_uav32():
    dest = fetch_uav32()
    for f in sorted((dest / "public" / "droneAudio").glob("*.wav")):
        out_path = TEST_DIR / out_name("drone", "uav32", _uav32_label(f.stem), f.stem)
        if out_path.exists():
            continue
        audio, sr = sf.read(str(f), dtype="float64")
        write_wav(out_path, audio, sr)


# --- summary ---------------------------------------------------------------

def print_summary():
    header = f"{'split':6} {'kind':10} {'source':10} {'label':18} {'files':>6} {'seconds':>10}"
    print(header)
    total_files = 0
    total_secs = 0.0
    for split_name, split_dir in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        groups = defaultdict(lambda: [0, 0.0])
        for f in sorted(split_dir.glob("*.wav")):
            info = parse_name(f.name)
            g = groups[(info["kind"], info["source"], info["label"])]
            g[0] += 1
            g[1] += sf.info(str(f)).duration
        for (kind, source, label), (n, secs) in sorted(groups.items()):
            print(f"{split_name:6} {kind:10} {source:10} {label:18} {n:6d} {secs:10.1f}")
            total_files += n
            total_secs += secs
    print(f"TOTAL: {total_files} files, {total_secs:.1f} seconds")


def main():
    if "--self-check" in sys.argv:
        _self_check()
        return
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    process_alemadi()
    process_svanstrom()
    process_esc50()
    process_dronenoise()
    process_uav32()
    print_summary()


if __name__ == "__main__":
    main()
