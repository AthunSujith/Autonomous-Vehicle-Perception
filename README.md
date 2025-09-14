# 🚘 Autonomous Vehicle Perception with YOLOv8  

This project builds a **YOLOv8-based perception system** using the **KITTI dataset**.  
It can detect **cars, pedestrians, and cyclists** in images, videos, and live streams.  

---

## 📂 Project Structure
```
Autonomous-Vehicle-Perception/
│
├── data/                       # Dataset folder (processed in YOLO format)
│   ├── raw/                    # Original KITTI dataset
│   ├── processed/              # YOLO-ready data
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── kitti.yaml          # Dataset config for YOLO
│
├── runs/                       # YOLO training outputs (weights, logs, results)
│
├── prepare_data.py             # Convert KITTI → YOLO format
├── register_coco_dataset.py    # Register dataset for COCO-style evaluation
├── utils.py                    # Helper functions (dataset checks, stats)
├── demo_notebook.ipynb         # Step-by-step training & inference notebook
├── app_media_mode.py           # Streamlit app (image/video/live detection)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📊 Dataset Source
- We use the **[KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)**.  
- Classes included:  
  - 🚗 Car  
  - 🚶 Pedestrian  
  - 🚴 Cyclist  

---

## ⚙️ Training the Model  

1️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```

2️⃣ **Prepare dataset (convert KITTI → YOLO format)**  
```bash
python prepare_data.py
```

3️⃣ **Train YOLOv8**  
```bash
yolo task=detect mode=train model=yolov8n.pt data=data/processed/kitti.yaml epochs=50 imgsz=640 batch=16 device=0
```

4️⃣ **Export trained model (optional)**  
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

---

## 🚀 Running the Demo App  

1️⃣ Run Streamlit app:  
```bash
streamlit run app_media_mode.py
```

2️⃣ Choose input mode in UI:  
- 🖼 Upload Image → detect cars, pedestrians, cyclists.  
- 🎞 Upload Video → run detection frame-by-frame.  
- 📷 Live Webcam → real-time detection.  

---

## 📈 Results & Use-Cases
- Detects **objects in driving scenes** from KITTI dataset.  
- Useful for:  
  - Autonomous driving research.  
  - Traffic surveillance.  
  - Pedestrian safety systems.  

---

✅ With this setup, you can **train, evaluate, and deploy** a computer vision system for self-driving cars in one project.  
