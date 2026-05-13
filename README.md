# AI-Powered Automatic Retail Checkout System

> Deep Neural Networks Semester Project — Varisha & Muzzammil

An end-to-end AI checkout system that uses **instance segmentation** (YOLOv8-seg / Mask R-CNN) to identify groceries on a checkout counter, generate itemised receipts, and manage inventory — all powered by a **FastAPI** backend and a **Flutter** POS interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Flutter](https://img.shields.io/badge/Flutter-3.16+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8--seg-Ultralytics-red)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Demo](#quick-demo)
4. [Setup Guide](#setup-guide)
5. [Dataset Preparation](#dataset-preparation)
6. [Model Training](#model-training)
7. [Backend API](#backend-api)
8. [Flutter Frontend](#flutter-frontend)
9. [Testing](#testing)
10. [Project Structure](#project-structure)

---

## Overview

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Model** | YOLOv8-seg / Mask R-CNN | Instance segmentation of 60 grocery categories |
| **Dataset** | MVTec D2S | 60 grocery categories, COCO JSON annotations |
| **Backend** | FastAPI + SQLite | Inference API, inventory management, transaction persistence |
| **Frontend** | Flutter + Riverpod 2.x | Cross-platform POS with segmentation overlays |
| **Database** | SQLite + SQLAlchemy | Products, transactions, inventory audit log |

---

## Architecture

```
Camera/Image → Flutter App → FastAPI Backend → YOLOv8-seg → Detection Results
                                    ↓
                              SQLite Database
                              (Products, Transactions, Inventory Log)
                                    ↓
                         JSON Response → Flutter UI
                         (Masks, Receipt, Totals)
```

---

## Quick Demo

### 1. Start the Backend

```bash
cd backend
pip install -r requirements.txt
python database/seed.py          # Seed 60 products into SQLite
python main.py                   # Start FastAPI on http://localhost:8000
```

Visit **http://localhost:8000/docs** for interactive Swagger UI.

### 2. Run the Flutter App

```bash
cd frontend
flutter pub get
flutter run -d chrome             # Web
flutter run                       # Android/iOS
```

### 3. Test Without a Camera

In the Flutter app, tap the **"Demo"** button to send a pre-loaded sample image to the backend.

---

## Setup Guide

### Prerequisites

- Python 3.10+
- Flutter SDK 3.16+ / Dart 3.2+
- CUDA 11.8+ (for GPU training, optional for inference)

### Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Seed the database
python database/seed.py

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Flutter Setup

```bash
cd frontend
flutter pub get
flutter run -d chrome
```

For Android emulator: backend URL defaults to `http://10.0.2.2:8000`
For physical device: update `lib/constants.dart` with your machine's LAN IP.

---

## Dataset Preparation

See [DATASET_PREP.md](DATASET_PREP.md) for detailed instructions.

```bash
# Convert MVTec D2S from COCO JSON to YOLO format
python data_preparation.py --d2s-root ./mvtec_d2s --output ./mvtec_d2s_yolo
```

---

## Model Training

### YOLOv8-seg (Primary)

```bash
python train.py --model yolo --data d2s.yaml --epochs 100 --batch 16 --device 0
```

### Mask R-CNN (Alternative)

```bash
python train.py --model maskrcnn --data d2s.yaml --epochs 100 --batch 2 --device 0
```

### Google Colab

Open `train_colab.ipynb` in Google Colab for GPU-accelerated training.

### Validation

```bash
python validate.py --model yolo --weights runs/segment/train/weights/best.pt --data d2s.yaml
```

### Single Image Prediction

```bash
python predict.py --image test.jpg --model yolo --weights runs/segment/train/weights/best.pt
```

After training, copy `best.pt` to `backend/weights/best.pt`.

---

## Backend API

See [API_DOCS.md](API_DOCS.md) for full endpoint documentation.

| Method | Endpoint | Description |
|--------|---------|-------------|
| POST | `/api/v1/checkout/scan` | Scan image for checkout |
| GET | `/api/v1/checkout/history` | Transaction history |
| GET | `/api/v1/checkout/debug/last` | Last processed image with masks |
| GET | `/api/v1/inventory/products` | List all products |
| GET | `/api/v1/inventory/products/{id}` | Get single product |
| PUT | `/api/v1/inventory/products/{id}` | Update product |
| POST | `/api/v1/inventory/restock` | Bulk restock |
| GET | `/api/v1/health` | Health check |

---

## Flutter Frontend

The Flutter app targets **Android, iOS, and Web** with four main screens:

1. **Checkout Screen** — Camera/image capture with segmentation overlay and receipt panel
2. **Receipt Screen** — Thermal receipt style with PDF print/share
3. **Inventory Screen** — Product management with color-coded stock levels
4. **History Screen** — Paginated transaction history with expandable details

---

## Testing

```bash
cd backend
python -m pytest tests/ -v
```

---

## Project Structure

```
dnn project/
├── backend/
│   ├── main.py                    # FastAPI entrypoint
│   ├── routers/
│   │   ├── checkout.py            # Checkout scan & history
│   │   └── inventory.py           # Product CRUD & restock
│   ├── models/
│   │   ├── inference.py           # Singleton model engine
│   │   └── schemas.py             # Pydantic models
│   ├── database/
│   │   ├── db.py                  # SQLAlchemy ORM
│   │   └── seed.py                # 60-product seed script
│   ├── utils/
│   │   └── item_counter.py        # Item counting utility
│   ├── tests/
│   │   ├── test_inference.py      # 20 inference tests
│   │   └── test_api.py            # 20 API tests
│   ├── weights/                   # Trained model weights
│   ├── .env                       # Configuration
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── lib/
│   │   ├── main.dart
│   │   ├── app.dart
│   │   ├── constants.dart
│   │   ├── screens/               # 4 screens
│   │   ├── widgets/               # Overlay, receipt tile, camera
│   │   ├── models/                # Dart data models
│   │   ├── services/              # API service
│   │   └── providers/             # Riverpod state
│   └── pubspec.yaml
├── data_preparation.py            # COCO → YOLO converter
├── d2s.yaml                       # Dataset config
├── train.py                       # Unified training script
├── validate.py                    # COCO metrics evaluation
├── predict.py                     # Single-image inference
├── train_colab.ipynb              # Google Colab notebook
├── requirements_training.txt
├── README.md
├── API_DOCS.md
├── ARCHITECTURE.md
└── DATASET_PREP.md
```
