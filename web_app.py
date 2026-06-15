"""Lokalny frontend do rejestracji, weryfikacji i identyfikacji twarzy."""
from threading import Lock
from typing import Iterable

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from config import COS_DIST_THRESHOLD, DB_PATH, IMG_EXTS
from pipeline import authenticate, quality
from pipeline.matcher import add_user, identify as identify_match
from pipeline.matcher import load_db, save_db, verify as verify_match

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 24 * 1024 * 1024

_db_lock = Lock()


def _json_error(message, status=400, **extra):
    return jsonify({"ok": False, "message": message, **extra}), status


def _valid_image_name(filename: str) -> bool:
    return bool(filename) and any(filename.lower().endswith(ext) for ext in IMG_EXTS)


def _decode_upload(file_storage):
    filename = secure_filename(file_storage.filename or "")
    if not _valid_image_name(filename):
        raise ValueError("Obsługiwane formaty: jpg, jpeg, png, bmp.")

    data = file_storage.read()
    if not data:
        raise ValueError("Przesłany plik jest pusty.")

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Nie udało się odczytać obrazu.")
    return img, filename


def _pipeline_cfg():
    no_tricks = request.form.get("no_tricks") == "on"
    if no_tricks:
        return authenticate.PipelineConfig(
            use_select=False,
            use_quality=False,
            use_restore=False,
            use_tta=False,
        )
    return authenticate.PipelineConfig()


def _embedding_from_image(img_bgr, cfg):
    result = authenticate.process(img_bgr, cfg)
    if result.embedding is None:
        reason = result.info.get("rejected", "pipeline odrzucił próbkę")
        raise ValueError(reason)
    threshold = (
        quality.adaptive_threshold(COS_DIST_THRESHOLD, result.quality)
        if cfg.use_quality
        else COS_DIST_THRESHOLD
    )
    return result, threshold


def _user_summary(db):
    return [
        {
            "name": user,
            "embeddings": len(record.get("embeddings", [])),
            "meta": record.get("meta", {}),
        }
        for user, record in sorted(db.items())
    ]


def _files(field: str) -> Iterable:
    return [file for file in request.files.getlist(field) if file and file.filename]


@app.get("/")
def index():
    db = load_db()
    return render_template("index.html", users=_user_summary(db), db_path=DB_PATH)


@app.get("/api/users")
def users():
    return jsonify({"ok": True, "users": _user_summary(load_db())})


@app.post("/api/enroll")
def enroll_user():
    username = secure_filename(request.form.get("username", "").strip())
    files = _files("images")

    if not username:
        return _json_error("Podaj identyfikator użytkownika.")
    if not files:
        return _json_error("Dodaj co najmniej jedno zdjęcie.")

    cfg = authenticate.PipelineConfig(
            use_select=False,
            use_quality=False,
            use_restore=False,
            use_tta=False,
        )
    
    embeddings = []
    accepted = []
    rejected = []

    for file_storage in files:
        original_name = file_storage.filename or "obraz"
        try:
            img, filename = _decode_upload(file_storage)
            result, _ = _embedding_from_image(img, cfg)
            embeddings.append(result.embedding)
            accepted.append(
                {
                    "file": filename,
                    "quality": round(float(result.quality), 4),
                    "info": result.info,
                }
            )
        except ValueError as exc:
            rejected.append({"file": original_name, "reason": str(exc)})

    if not embeddings:
        return _json_error("Żadne zdjęcie nie przeszło pipeline'u.", rejected=rejected)

    with _db_lock:
        db = load_db()
        if username in db:
            return _json_error(
                "Ten użytkownik już istnieje. Wybierz inną nazwę.",
                status=409,
                accepted=accepted,
                rejected=rejected,
            )
        add_user(
            db,
            username,
            embeddings,
            meta={
                "n_enroll": len(embeddings),
                "source": "web_frontend",
                "no_tricks": not cfg.use_quality,
            },
        )
        save_db(db)
        users_now = _user_summary(db)

    return jsonify(
        {
            "ok": True,
            "message": f"Zarejestrowano użytkownika {username}.",
            "user": username,
            "accepted": accepted,
            "rejected": rejected,
            "users": users_now,
        }
    )


@app.post("/api/verify")
def verify_user():
    username = request.form.get("username", "").strip()
    files = _files("image")

    if not username:
        return _json_error("Wybierz użytkownika do weryfikacji.")
    if not files:
        return _json_error("Dodaj zdjęcie do weryfikacji.")

    try:
        img, filename = _decode_upload(files[0])
        cfg = _pipeline_cfg()
        result, threshold = _embedding_from_image(img, cfg)
    except ValueError as exc:
        return _json_error(str(exc))

    db = load_db()
    accepted, distance = verify_match(db, username, result.embedding, threshold=threshold)
    return jsonify(
        {
            "ok": True,
            "mode": "verify",
            "file": filename,
            "user": username,
            "accepted": bool(accepted),
            "distance": round(float(distance), 4),
            "threshold": round(float(threshold), 4),
            "quality": round(float(result.quality), 4),
            "info": result.info,
            "message": "Akceptacja" if accepted else "Odrzucenie",
        }
    )


@app.post("/api/identify")
def identify_user():
    files = _files("image")
    if not files:
        return _json_error("Dodaj zdjęcie do identyfikacji.")

    try:
        img, filename = _decode_upload(files[0])
        cfg = _pipeline_cfg()
        result, threshold = _embedding_from_image(img, cfg)
    except ValueError as exc:
        return _json_error(str(exc))

    db = load_db()
    predicted_user, ranking = identify_match(db, result.embedding, threshold=threshold)
    return jsonify(
        {
            "ok": True,
            "mode": "identify",
            "file": filename,
            "predicted_user": predicted_user,
            "ranking": [
                {"user": user, "distance": round(float(distance), 4)}
                for user, distance in ranking
            ],
            "threshold": round(float(threshold), 4),
            "quality": round(float(result.quality), 4),
            "info": result.info,
            "message": predicted_user or "Brak dopasowania w bazie",
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
