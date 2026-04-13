# Revly

**Own your monetization.** Revly turns any object in your live stream into a native ad placement — powered by real-time computer vision, running natively inside OBS. Streamers keep 90% of the revenue.

> Winner ($300) at [BagelHacks II](https://bagelhacks-ii.devpost.com/) | [DevPost](https://devpost.com/software/revly) | [Landing Page](https://landing-phi-jade-28.vercel.app)

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Node.js](https://img.shields.io/badge/Node.js-18+-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple) ![License](https://img.shields.io/badge/License-ISC-lightgrey)

---

## How It Works

Revly detects real objects in your stream (cups, phones, laptops, bottles, books) and seamlessly replaces them with sponsor ad placements in real time. Your coffee cup becomes a branded product. The phone on your desk becomes a sponsor moment. No ad breaks, no viewer interruption.

```
Camera Feed → YOLOv8 Detection → Kalman Tracking → Ad Replacement → OBS Output
                                       ↓
                              Smart Timing Engine
                         (silence, scene change, motion)
```

### Three Deployment Modes

| Mode | How it works | Best for |
|------|-------------|----------|
| **Overlay** | Positions ad images over detected objects as OBS sources | Simple setup, lowest latency |
| **Replace** | Pixel-level object replacement streamed via MJPEG | Seamless blending, Poisson compositing |
| **AI** | Buffers stream → generative AI (Wan 2.7) → re-injects edited footage | Photorealistic product placement |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                    OBS Studio                    │
│  ┌──────────────────────────────────────────┐   │
│  │         ad_stream_plugin.py              │   │
│  │  (OBS Python Script — main entry point)  │   │
│  └──────────┬───────────────────────────────┘   │
└─────────────┼───────────────────────────────────┘
              │
   ┌──────────▼──────────┐
   │    lib/ modules      │
   │  ├── detector.py     │  YOLOv8 + FastSAM (threaded inference)
   │  ├── tracker.py      │  Kalman filter & EMA smoothing
   │  ├── replacer.py     │  Perspective warp, Poisson blend, color adapt
   │  ├── timing.py       │  Silence, scene change, motion detection
   │  ├── overlay.py      │  OBS source management
   │  ├── virtual_cam.py  │  MJPEG stream output
   │  ├── frame_capture.py│  Camera/screen capture
   │  └── config.py       │  Configuration & class mappings
   └──────────┬──────────┘
              │ Socket.IO
   ┌──────────▼──────────┐
   │  frontend/server.js  │  Node.js + Express + Socket.IO
   │  ├── dashboard.html  │  Creator control panel
   │  ├── overlay.html    │  Browser source overlay
   │  └── /api/*          │  REST + HTTP triggers (Stream Deck)
   └─────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- OBS Studio (for plugin mode) or just a webcam (for demo mode)

### 1. Install Python Dependencies

```bash
# Option A: automated installer
chmod +x install.sh && ./install.sh

# Option B: manual
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies

```bash
cd frontend
npm install
```

### 3. Run

**Standalone Demo (no OBS needed):**

```bash
python demo.py                       # webcam + detection overlay
python demo.py --ad assets/cocacola.png  # with custom ad image
python demo.py --stream              # also starts MJPEG stream at localhost:8765
python demo.py --sam                 # enable FastSAM for precise masks
```

**Full Stack (Dashboard + Detection Bridge):**

```bash
# Terminal 1: Start the overlay server
cd frontend && node server.js

# Terminal 2: Start the detection bridge
python detection_bridge.py
```

Then open:
- **Dashboard:** http://localhost:3000/dashboard.html
- **Overlay (add as OBS Browser Source):** http://localhost:3000/overlay.html?token=YOUR_TOKEN

**OBS Plugin Mode:**

1. Open OBS Studio → Tools → Scripts
2. Click `+` and select `ad_stream_plugin.py`
3. Configure settings in the script properties panel
4. Place ad images in `assets/`
5. Click **Start Processing**

---

## Project Structure

```
revly/
├── ad_stream_plugin.py      # OBS Python script plugin (main entry)
├── detection_bridge.py      # Camera → Socket.IO detection bridge
├── demo.py                  # Standalone demo (no OBS required)
├── install.sh               # Dependency installer
├── requirements.txt         # Python dependencies
│
├── lib/                     # Core ML & processing modules
│   ├── detector.py          # YOLOv8 + FastSAM threaded inference engine
│   ├── tracker.py           # Kalman filter & EMA bounding box smoothing
│   ├── replacer.py          # Perspective warp, Poisson blend, color adaptation
│   ├── timing.py            # Smart ad timing (silence, scene change, motion)
│   ├── overlay.py           # OBS overlay source management
│   ├── virtual_cam.py       # MJPEG stream output & replace mode processor
│   ├── frame_capture.py     # Camera & screen capture abstraction
│   └── config.py            # Configuration, class mappings, constants
│
├── frontend/                # Creator dashboard & overlay server
│   ├── server.js            # Express + Socket.IO server
│   └── public/
│       ├── dashboard.html   # Creator control panel
│       ├── dashboard.js     # Dashboard client logic
│       ├── overlay.html     # Browser source overlay for OBS
│       ├── overlay.js       # Overlay rendering engine
│       └── detection-renderer.js
│
├── config/
│   └── class_mapping.json   # COCO class ID → sponsor image mapping
│
├── assets/                  # Ad images and test outputs
│   └── cocacola.png         # Example sponsor image
│
└── landing/                 # Landing page (deployed on Vercel)
```

---

## Features

### Computer Vision Pipeline
- **YOLOv8** object detection at 30+ FPS on consumer hardware
- **FastSAM** segmentation for pixel-accurate masks
- **Kalman filter** + EMA smoothing for rock-stable bounding boxes (no flicker)
- **Poisson blending** for seamless compositing
- **Perspective-aware warping** and automatic **color adaptation**

### Smart Timing Engine
- **Audio silence detection** — ads appear during natural pauses
- **Scene change analysis** — avoids disrupting high-energy moments
- **Motion tracking** — holds ads during action, places during calm
- **Configurable cooldowns** — never over-serve placements

### Creator Dashboard
- Real-time ad library management with drag-and-drop uploads
- Preset positioning templates (corner, lower-third, banner, popup, ticker, takeover)
- Live sync across devices via Socket.IO
- HTTP trigger endpoints for Stream Deck / OBS hotkey integration
- Ad rotation with configurable intervals

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/token` | Generate overlay token |
| `GET /api/trigger/:token/show/:adId` | Show specific ad |
| `GET /api/trigger/:token/hide/:adId` | Hide specific ad |
| `GET /api/trigger/:token/toggle/:adId` | Toggle ad visibility |
| `GET /api/trigger/:token/hide-all` | Panic — hide all ads |
| `GET /api/trigger/:token/show-by-index/:n` | Toggle ad by position (1-based) |
| `POST /api/upload` | Upload ad image (multipart) |

---

## Configuration

### Class Mapping (`config/class_mapping.json`)

Maps COCO class IDs to sponsor images. Detected objects matching these classes get replaced:

```json
{
  "41": "cocacola.png",
  "39": "cocacola.png",
  "67": "cocacola.png",
  "63": "cocacola.png",
  "73": "cocacola.png"
}
```

| Class ID | Object |
|----------|--------|
| 39 | Bottle |
| 41 | Cup |
| 63 | Laptop |
| 67 | Cell phone |
| 73 | Book |

### Plugin Settings (OBS or `lib/config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `mode` | `overlay` | `overlay` or `replace` |
| `model_name` | `yolov8n.pt` | YOLO model (nano/small/medium) |
| `confidence_threshold` | `0.3` | Detection confidence cutoff |
| `smoothing_method` | `kalman` | `kalman` or `ema` |
| `blend_mode` | `alpha` | `alpha` or `poisson` |
| `min_ad_interval` | `60s` | Minimum seconds between ad placements |
| `min_ad_duration` | `5s` | Minimum ad display time |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Segmentation | FastSAM |
| Tracking | Kalman Filter, EMA |
| Compositing | OpenCV, Poisson Blending, NumPy |
| Generative AI | Alibaba Wan 2.7 (DashScope) |
| Streaming Server | Node.js, Express, Socket.IO |
| OBS Integration | obspython |
| Real-time Comms | Socket.IO, MJPEG |

---

## Team

| | Name | GitHub |
|---|------|--------|
| | Bohan Wang | [@bohan-wang](https://devpost.com/bohan-wang) |
| | Yueran Lu | [@yueranlu05](https://devpost.com/yueranlu05) |
| | Sivabalan Muthurajan | [@sivabalansm](https://github.com/sivabalansm) |
| | Zayan Khan | [@zk9810380](https://devpost.com/zk9810380) |

---

## License

ISC
