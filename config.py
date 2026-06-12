"""Globalna konfiguracja: ścieżki, stałe, seed.

Projekt 3 (Biometria) — uwierzytelnianie in-the-wild.
Dwa utrudnienia: #3 niska rozdzielczość (CCTV) + #1 wiele twarzy / plakaty.
Backbone ArcFace iresnet50 używany BEZ dotrenowywania — odporność budujemy trikami.
"""
from pathlib import Path

SEED = 42
EMBEDDING_DIM = 512
IMG_SIZE = 112  # rozmiar wejścia ArcFace

# --- katalogi -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# Dane reużywane z części 1 (folder obok). Tożsamości FaceCard + własne twarze grupy.
SIBLING = ROOT.parent / "face-recognition_"
FACES_TRAINING = SIBLING / "faces_training"   # 427 tożsamości FaceCard (LFW-style)
NEW_FACES = SIBLING / "new_faces"             # 100 dodatkowych tożsamości
OWN_FACES = SIBLING / "own_faces"             # asia / maria / michal (członkowie grupy)

# Backbone — leży lokalnie w tym folderze.
BACKBONE_PATH = ROOT / "ms1mv3_arcface_r50.pth"

# --- artefakty generowane -------------------------------------------------
ARTIFACTS = ROOT / "artifacts"
DB_PATH = ARTIFACTS / "registered_users.pkl"   # baza enrollmentu (multi-embedding/os.)
SPLIT_PATH = ARTIFACTS / "data_split.json"      # podział tożsamości na role
DATA_DIR = ARTIFACTS / "data"                   # wygenerowane zbiory testowe
CLEAN_DIR = DATA_DIR / "clean"
LOWRES_DIR = DATA_DIR / "lowres"
MULTIFACE_DIR = DATA_DIR / "multiface"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"
RESULTS_DIR = ARTIFACTS / "results"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# --- progi / parametry decyzyjne -----------------------------------------
# Próg dystansu kosinusowego dobierany na walidacji (Exp kalibracji).
# Wartość startowa; nadpisywana przez wynik kalibracji.
COS_DIST_THRESHOLD = 0.80

for _d in (ARTIFACTS, DATA_DIR, CLEAN_DIR, LOWRES_DIR, MULTIFACE_DIR,
           BACKGROUNDS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
