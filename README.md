# ClearRoute 🗑️🚚

Street-litter detection in Konstanz (video + YOLO) and a **dashboard** for the urban
cleaning service, with **AI-assisted collection route planning**.

---

## Project structure

```
clearroute/
├── app.py                    ← Streamlit dashboard (map, priority, ROUTES)
├── routing.py                ← Route planning (Claude + Google Maps)
├── detect.py                 ← YOLO over video → data/dados_reais_*.json
├── webcam_teste.py           ← Real-time YOLO webcam test
├── treino_clearroute.ipynb   ← Model training notebook (Colab)
├── data/
│   ├── data_example.json     ← Example detections (demo)
│   └── data_real.json        ← Example real output
├── videos/                   ← Example input videos
├── models/                   ← (gitignored) trained best.pt weights
├── frames/                   ← (gitignored) annotated frames produced by detect.py
├── requirements.txt
└── .streamlit/
    ├── secrets.toml          ← (gitignored) your API keys
    └── secrets.toml.example  ← template
```

---

## Install

```bash
pip install -r requirements.txt
```

---

## The dashboard

```bash
streamlit run app.py        # run it from INSIDE the clearroute/ folder
```

Open `http://localhost:8501`. Sections:

1. **Detection map** — heatmap + priority markers over Konstanz.
2. **Priority queue** — sorted by `score = confidence × type weight`.
3. **🚚 Route planning** *(new)* — set how many vehicles are available and generate the
   most efficient route for each one (see below).
4. **Detail view** — inspect each detection (with annotated frame if available).

Works with `data/data_example.json` without needing the model.

---

## Route planning (AI + Google Maps)

Given the detected coordinates and the number of vehicles, the system generates the
most efficient route for each one. All vehicles start and end at the base:
**Fritz-Arnold-Straße 2B, 78467 Konstanz-Industriegebiet**.

Pipeline (hybrid):

1. **Claude (`claude-opus-4-8`)** assigns each point to a vehicle and orders the stops
   (balances load, groups by proximity and prioritises the highest-`score` points).
2. **Google Directions** computes the real road path and travel time.
3. **Google Maps** draws one coloured route per vehicle, with the depot and numbered
   stops. An **OpenStreetMap** view is also available as a free fallback.

### Required API keys

Copy the template and fill in your keys:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

| Key                   | What it's for                     | Where to get it |
|-----------------------|-----------------------------------|------------------|
| `ANTHROPIC_API_KEY`   | AI that generates the routes (Claude) | console.anthropic.com → API Keys |
| `GOOGLE_MAPS_API_KEY` | Map + real road routes            | console.cloud.google.com |

In Google Cloud enable: **Maps JavaScript API**, **Directions API** and **Geocoding API**
(billing must be active; there is a monthly free tier). It's good practice to
**restrict** the key by HTTP referrer.

> **Without keys** the dashboard still works as a preview: it uses a local assignment
> (k-means + nearest neighbour, no AI) and an OpenStreetMap map.

---

## Generate real data (detection)

`detect.py` runs the trained model over a video and saves the detections.
It needs the weights in `models/best.pt` (not included — trained with
`treino_clearroute.ipynb` and the *Trash-AI v2* dataset from Roboflow) and you must
edit `VIDEO_PATH`:

```bash
python detect.py            # generates data/dados_reais_<timestamp>.json + frames/
```

Detected litter classes: `Flasche, Dose, Karton, Becher, Maske, Nadel, Papier,
Plastik, Müll`.

---

## Detection JSON fields

| Field         | Type   | Description                                   |
|---------------|--------|-----------------------------------------------|
| `lat`         | number | Latitude                                      |
| `lon`         | number | Longitude                                     |
| `typ`         | text   | Litter type                                   |
| `konfidenz`   | number | Model confidence (0–1)                        |
| `zeitstempel` | text   | Moment in the video (MM:SS)                   |
| `frame_path`  | text   | (optional) path to the annotated frame (real data) |

---

## Tech stack
- Python 3.10+
- [Streamlit](https://streamlit.io/) · [Folium](https://python-visualization.github.io/folium/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Claude API](https://docs.claude.com/) (`claude-opus-4-8`)
- [Google Maps Platform](https://developers.google.com/maps) (Maps JS, Directions, Geocoding)

---

## Next improvements
- Real per-frame GPS in `detect.py` (it currently uses fixed time segments).
- VRP constraints: per-vehicle capacity and time windows.
- At >500 points: a dedicated solver (OR-Tools) or Google's Route Optimization API.
- Route persistence (history) and token-cost estimation.
