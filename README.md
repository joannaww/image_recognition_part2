# Biometria — Projekt 3: uwierzytelnianie twarzą *in-the-wild*

System uwierzytelniania (weryfikacja 1:1 + identyfikacja 1:N) odporny na **dwa
utrudnienia z trudnymi danymi**:

- **#3 niska rozdzielczość / jakość (CCTV)** — restauracja twarzy, FIQA, adaptacyjny próg, TTA
- **#1 wiele twarzy na zdjęciu** — detekcja wszystkich twarzy + polityka selekcji właściwej
  osoby (duża, centralna, pewnie wykryta) i odrzucanie przypadków niejednoznacznych

**Założenie projektu:** backbone ArcFace `iresnet50` (`ms1mv3_arcface_r50.pth`) jest
**zamrożony — bez dotrenowywania**. Odporność budujemy *trikami test-time*. Cel:
FRR na próbkach trudnych nie rośnie o >5 pp, TIR nie spada o >5 pp; FAR minimalizujemy.

## Pipeline (jeden łańcuch, każdy trik z flagą do ablacji)

```
obraz → detekcja(wszystkie twarze) → selekcja właściwej(#1)
      → FIQA(#3) → restauracja warunkowa(#3) → align 112² → embedding(+TTA)
      → matcher (galeria multi-embedding, dystans kosinusowy, próg adaptacyjny)
```

Konfiguracja trików: `pipeline/authenticate.py::PipelineConfig`.

## Środowisko

Używamy gotowego venv z części 1 (ma torch/insightface/opencv):

```bash
PY=../face-recognition_/env/bin/python      # albo: python -m venv env && pip install -r requirements.txt
```

Detektor InsightFace `buffalo_l` dociąga się sam przy pierwszym uruchomieniu (CPU).

## Uruchomienie (pipeline danych)

```bash
$PY build_split.py                 # podział tożsamości → artifacts/data_split.json
$PY enroll.py --per-user 3         # wdrożenie 100 użytkowników → registered_users.pkl
$PY build_test_sets.py --lowres-per 6 --multiface-per 6   # zbiory trudne + manifest.json
```

Pojedyncza próbka:

```bash
$PY verify.py <obraz> <user>       # 1:1   (--no-tricks = baseline bez trików)
$PY identify.py <obraz>            # 1:N
```

## Dane

Reużywane z `../face-recognition_/`: `faces_training` (427 tożsamości FaceCard),
`own_faces` (członkowie grupy: asia/maria/michal). Role tożsamości (enrolled /
impostors / distractors) ustala `build_split.py`.

- **clean** — held-out czyste zdjęcia wdrożonych + impostorzy
- **lowres** (#3) — syntetyczna degradacja CCTV tych samych zdjęć → *sparowane* czyste↔trudne
- **multiface** (#1) — kompozyty: subiekt (duży, centralny) + 1–3 dystraktory na tle

> **Uwaga:** zdjęcia FaceCard to ciasne cropy twarzy — detektor często ich nie
> widzi w oryginale (enrollment używa wtedy center-crop fallback). W kompozytach
> doklejane twarze są paddowane (replikacja brzegu), żeby były wykrywalne.

## Status / dalsze kroki

Gotowe: backbone + pełny pipeline detekcji/selekcji/jakości/embeddingu, enrollment
100 osób, generator zbiorów trudnych. **Do zrobienia:**

- ewaluacja: `evaluate.py` / notebooki — EER, FAR/FRR, TIR, kalibracja progu na walidacji,
  ablacje trików, porównanie clean vs trudne (cel ≤5 pp). **Najpierw zmierzyć baseline**
  (ArcFace bez trików) na zbiorach trudnych — to pokaże, ile realnie trzeba nadrobić.
- #3: jeśli baseline na lowres spada >5 pp — podmiana restauracji na **GFPGAN/CodeFormer**
  (`pipeline/restore.py`); inaczej wystarczy baseline + FIQA + adaptacyjny próg.
- #1: dostrojenie polityki selekcji i progu niejednoznaczności na podstawie wyników multiface.
