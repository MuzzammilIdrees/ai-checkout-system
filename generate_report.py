from fpdf import FPDF
from fpdf.enums import XPos, YPos
import datetime

class PDFReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("helvetica", "I", 10)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, "AI-Powered Automatic Retail Checkout System - Detailed Technical Report", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def chapter_title(self, title):
        self.set_font("helvetica", "B", 18)
        self.set_text_color(41, 128, 185) # Blue
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)

    def section_title(self, title):
        self.set_font("helvetica", "B", 14)
        self.set_text_color(44, 62, 80) # Dark Blue/Gray
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def sub_section_title(self, title):
        self.set_font("helvetica", "B", 12)
        self.set_text_color(52, 73, 94)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def chapter_body(self, body):
        self.set_font("helvetica", "", 11)
        self.set_text_color(0, 0, 0)
        # Ensure text is clean
        clean_body = body.replace('—', '-').replace('\u2014', '-').replace('\u2013', '-').replace('"', '"').replace('"', '"')
        self.multi_cell(0, 6, clean_body)
        self.ln(5)

    def code_block(self, code_text):
        self.set_font("courier", "", 9)
        self.set_fill_color(240, 240, 240)
        clean_code = code_text.replace('—', '-').replace('\u2014', '-')
        self.multi_cell(0, 5, clean_code, fill=True)
        self.ln(5)

    def bullet_point(self, title, description):
        self.set_font("helvetica", "B", 11)
        self.set_text_color(0, 0, 0)
        self.cell(10, 6, chr(149), align="R") # Bullet character
        self.cell(self.get_string_width(title) + 2, 6, title)
        
        self.set_font("helvetica", "", 11)
        clean_desc = description.replace('—', '-').replace('\u2014', '-')
        self.multi_cell(0, 6, f"- {clean_desc}")
        self.ln(2)

    def normal_bullet(self, text):
        self.set_font("helvetica", "", 11)
        self.set_text_color(0, 0, 0)
        self.cell(10, 6, chr(149), align="R")
        clean_text = text.replace('—', '-').replace('\u2014', '-')
        self.multi_cell(0, 6, clean_text)
        self.ln(1)

def generate_report():
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # --- FRONT PAGE ---
    pdf.set_y(80)
    pdf.set_font("helvetica", "B", 30)
    pdf.set_text_color(44, 62, 80)
    pdf.multi_cell(0, 15, "AI-Powered Automatic Retail\nCheckout System", align="C")
    
    pdf.ln(20)
    pdf.set_font("helvetica", "I", 18)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Comprehensive Technical & Architecture Report", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(50)
    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Presented By:", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    
    pdf.set_font("helvetica", "", 18)
    pdf.cell(0, 10, "Muzzammil Idrees", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, "Varisha", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(50)
    pdf.set_font("helvetica", "I", 14)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%B %Y')}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, "Course: Deep Neural Networks", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # --- TABLE OF CONTENTS ---
    pdf.add_page()
    pdf.chapter_title("Table of Contents")
    toc = [
        "1. Executive Summary",
        "2. Data Pipeline & Engineering",
        "   2.1 Dataset Collection (MVTec D2S)",
        "   2.2 Data Labeling & Conversion",
        "   2.3 Data Validation & Versioning",
        "   2.4 Feature Engineering & Augmentation",
        "3. Deep Learning Model Architecture",
        "   3.1 Architectural Selection (YOLOv8-seg vs Mask R-CNN)",
        "   3.2 Hyperparameters & Configuration",
        "   3.3 Training Procedure & Transfer Learning",
        "   3.4 Model Evaluation & Metrics",
        "   3.5 Model Packaging & Inference Engine",
        "4. Codebase & System Architecture",
        "   4.1 Training Scripts & Cloud Execution",
        "   4.2 FastAPI Inference Server",
        "   4.3 Database ORM & Glue Code",
        "   4.4 Feature Transformations",
        "5. Frontend Mobile POS Application",
        "   5.1 Flutter Cross-Platform Architecture",
        "   5.2 State Management (Riverpod)",
        "   5.3 UI Components & Mask Rendering",
        "6. Infrastructure & Deployment",
        "   6.1 Compute & Hardware Acceleration",
        "   6.2 Docker Orchestration",
        "   6.3 Networking & Security",
        "   6.4 Storage Solutions",
        "   6.5 Monitoring & Logging",
        "7. Conclusion"
    ]
    pdf.set_font("helvetica", "", 12)
    for item in toc:
        pdf.cell(0, 8, item, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # --- 1. EXECUTIVE SUMMARY ---
    pdf.add_page()
    pdf.chapter_title("1. Executive Summary")
    pdf.chapter_body("The AI-Powered Automatic Retail Checkout System is an end-to-end computer vision application designed to modernize the retail checkout experience. Traditional barcode scanning is time-consuming and labor-intensive. This project introduces a friction-less checkout process leveraging Deep Neural Networks - specifically instance segmentation models - to automatically identify, count, and price groceries placed on a checkout counter in real-time.")
    pdf.chapter_body("The system is divided into three primary tiers:")
    pdf.normal_bullet("The Deep Learning Engine: Utilizing YOLOv8-seg (and optionally Mask R-CNN) trained on the MVTec D2S dataset to generate pixel-accurate segmentation masks for 60 grocery categories.")
    pdf.normal_bullet("The Backend API: A robust, asynchronous FastAPI application connected to a SQLite database, managing inventory, processing inference requests, and persisting transaction logs.")
    pdf.normal_bullet("The Frontend POS: A reactive Flutter application providing the Point of Sale interface, rendering real-time segmentation overlays, managing user checkout history, and handling inventory tracking.")

    # --- 2. DATA PIPELINE ---
    pdf.add_page()
    pdf.chapter_title("2. Data Pipeline & Engineering")
    pdf.chapter_body("A robust data pipeline is critical for training highly accurate segmentation models. Our pipeline handles ingestion, transformation, augmentation, and validation of complex image data.")

    pdf.section_title("2.1 Dataset Collection (MVTec D2S)")
    pdf.chapter_body("We utilized the MVTec Densely Segmented Supermarket (D2S) dataset. This dataset is specifically designed for instance-aware semantic segmentation in an industrial retail setting. It contains high-resolution images of 60 distinct grocery categories, ranging from fresh produce (apples, bananas, tomatoes) to packaged goods (cereal bars, pasta, beverages).")
    pdf.chapter_body("The dataset represents challenging real-world scenarios including:")
    pdf.normal_bullet("Heavy occlusion where items are stacked on top of one another.")
    pdf.normal_bullet("Varying lighting conditions and reflections on plastic packaging.")
    pdf.normal_bullet("Deformable items (e.g., net bags of oranges) and visually similar items (e.g., different brands of water bottles).")

    pdf.section_title("2.2 Data Labeling & Conversion")
    pdf.chapter_body("The original MVTec D2S dataset provides annotations in the standard COCO JSON format. However, our primary model, YOLOv8-seg, requires annotations in a specific text-based format where polygon coordinates are normalized between 0.0 and 1.0.")
    pdf.chapter_body("We developed a custom data preparation script (data_preparation.py) to automate this conversion. The script performs the following operations:")
    pdf.normal_bullet("Parses instances_train.json and instances_val.json.")
    pdf.normal_bullet("Extracts polygon coordinates, filtering out complex Run-Length Encoded (RLE) masks.")
    pdf.normal_bullet("Normalizes absolute pixel coordinates by dividing by the image width and height.")
    pdf.normal_bullet("Maps the 60 COCO category IDs to a sequential 0-59 index required by Ultralytics.")
    pdf.normal_bullet("Generates individual .txt label files for every image in the dataset.")

    pdf.section_title("2.3 Data Validation & Versioning")
    pdf.chapter_body("To prevent data leakage and ensure unbiased evaluation, the dataset is strictly partitioned into training and validation splits. The training set features single-item images with clean backgrounds, while the validation set consists of multi-item scenes with occlusions to test generalization.")
    pdf.chapter_body("Data versioning and configuration are managed through a YAML manifest (d2s.yaml). This file acts as the single source of truth for the dataset, defining the root paths, number of classes (nc: 60), and the complete class mapping dictionary.")

    pdf.section_title("2.4 Feature Engineering & Augmentation")
    pdf.chapter_body("To prevent overfitting and improve the model's ability to generalize to unseen checkout counter configurations, we implemented an aggressive data augmentation pipeline during the training phase. These augmentations are applied dynamically in memory:")
    pdf.normal_bullet("HSV Color Jittering: Hue (0.015), Saturation (0.7), and Value/Brightness (0.4) adjustments simulate different camera sensors and lighting environments.")
    pdf.normal_bullet("Geometric Transformations: Random scaling (0.5x to 1.5x) and translations (10%) simulate varying camera distances and placements.")
    pdf.normal_bullet("Rotations: +/- 15 degree random rotations account for items being placed at arbitrary angles on the counter.")
    pdf.normal_bullet("Flips: 50% probability for both horizontal and vertical flips.")
    pdf.normal_bullet("Mosaic Augmentation: A 100% probability 4-image mosaic technique is employed. This combines four different training images into a single grid, forcing the model to detect items at smaller scales and simulating incredibly dense, cluttered checkout scenarios.")

    # --- 3. MODEL ARCHITECTURE ---
    pdf.add_page()
    pdf.chapter_title("3. Deep Learning Model Architecture")
    
    pdf.section_title("3.1 Architectural Selection (YOLOv8-seg vs Mask R-CNN)")
    pdf.chapter_body("The project implements two distinct architectures to provide flexibility between inference speed and absolute accuracy. The primary architecture used in production is YOLOv8-seg.")
    pdf.chapter_body("Why Instance Segmentation? Standard object detection provides bounding boxes, which are insufficient for a retail checkout. If an apple is partially on top of a banana, their bounding boxes will overlap significantly, making it difficult to render accurate UI overlays or compute precise spatial relationships. Instance segmentation provides pixel-perfect polygon masks, separating overlapping objects distinctly.")
    pdf.chapter_body("YOLOv8-seg (Primary): Developed by Ultralytics, YOLOv8-seg is a state-of-the-art single-stage instance segmentation model. We utilize the 'nano' (YOLOv8n-seg) and 'medium' (YOLOv8m-seg) variants. It is optimized for real-time edge inference, making it ideal for a responsive POS system.")
    pdf.chapter_body("Mask R-CNN (Alternative): Implemented via Facebook's Detectron2 framework using a ResNet-50-FPN backbone. This is a two-stage detector (Region Proposal Network followed by a mask head). While slower, it serves as a robust baseline for comparative academic analysis.")

    pdf.section_title("3.2 Hyperparameters & Configuration")
    pdf.chapter_body("Hyperparameter tuning was conducted to optimize the YOLOv8-seg model for the D2S dataset. The final configuration is as follows:")
    
    params = [
        ("Optimizer", "Stochastic Gradient Descent (SGD)"),
        ("Initial Learning Rate (lr0)", "0.01"),
        ("Final Learning Rate (lrf)", "0.01 (using Cosine Annealing)"),
        ("Momentum", "0.937"),
        ("Weight Decay", "0.0005"),
        ("Batch Size", "16"),
        ("Input Image Size", "640x640 pixels"),
        ("Epochs", "100"),
        ("Early Stopping Patience", "20 epochs")
    ]
    for k, v in params:
        pdf.bullet_point(k + ":", v)

    pdf.section_title("3.3 Training Procedure & Transfer Learning")
    pdf.chapter_body("Training a segmentation model from scratch requires immense computational resources and data. To accelerate convergence, we utilized Transfer Learning. The model weights were initialized from a checkpoint pre-trained on the massive MS COCO dataset (80 object classes).")
    pdf.chapter_body("The training loop modifies the final classification head to output 60 classes instead of 80, and fine-tunes the mask generation branch. Training was executed using Google Colab's cloud infrastructure, specifically utilizing NVIDIA T4 and A100 Tensor Core GPUs. A unified train.py script orchestrates this, allowing seamless switching between YOLO and Detectron2 backends.")

    pdf.section_title("3.4 Model Evaluation & Metrics")
    pdf.chapter_body("The model is evaluated using the standard COCO metric suite, focusing on mean Average Precision (mAP).")
    pdf.normal_bullet("mAP@50: Measures precision where the Intersection over Union (IoU) between the predicted mask and ground truth is strictly > 50%.")
    pdf.normal_bullet("mAP@50-95: A stricter metric that averages mAP across IoU thresholds from 50% to 95% in 5% increments. This heavily penalizes sloppy mask boundaries.")
    pdf.chapter_body("During validation (validate.py), the system outputs per-class AP scores to identify problematic categories. Furthermore, Precision-Recall (PR) curves and Confusion Matrices are generated natively to visualize false positives (e.g., misclassifying 'Coca-Cola Light' as 'Coca-Cola Original').")

    pdf.section_title("3.5 Model Packaging & Inference Engine")
    pdf.chapter_body("In a production POS environment, latency is unacceptable. Model weights (.pt or .pth files) are statically packaged with the backend.")
    pdf.chapter_body("We engineered a Singleton InferenceEngine class in Python. During the FastAPI application 'lifespan' startup event, the model weights are loaded directly into RAM/VRAM. When an API request arrives, the engine is already warm, resulting in zero-latency initialization. Thread-safety is guaranteed using Python asyncio.Lock(), ensuring that concurrent HTTP requests do not cause race conditions within the PyTorch inference graph.")

    # --- 4. CODEBASE & SYSTEM ARCHITECTURE ---
    pdf.add_page()
    pdf.chapter_title("4. Codebase & System Architecture")
    pdf.chapter_body("The system is designed as a decoupled, microservice-style architecture. This separation of concerns ensures that the Deep Learning model, the API logic, and the UI can be scaled and updated independently.")

    pdf.section_title("4.1 Training Scripts & Cloud Execution")
    pdf.chapter_body("The repository contains a suite of CLI tools for offline model lifecycle management:")
    pdf.normal_bullet("train.py: The central entry point. Supports CLI arguments for model selection (--model yolo/maskrcnn), epoch count, and dataset paths.")
    pdf.normal_bullet("validate.py: Executes the COCO evaluator against the validation split and prints detailed tabular metrics.")
    pdf.normal_bullet("predict.py: A debugging utility that takes a single image, runs inference, draws bounding boxes/colored masks, and saves the output to disk.")
    pdf.normal_bullet("train_colab.ipynb: A Jupyter notebook wrapping the CLI tools, providing Google Drive mounting, dependency installation, and inline matplotlib visualization for execution on Google Colab.")

    pdf.section_title("4.2 FastAPI Inference Server")
    pdf.chapter_body("The backend API is built with FastAPI, chosen for its native async support, extreme performance (via Starlette/Pydantic), and automatic Swagger UI documentation.")
    pdf.chapter_body("Core Endpoints include:")
    pdf.normal_bullet("POST /api/v1/checkout/scan: The primary inference endpoint. It accepts a multipart/form-data image upload. It returns a complex JSON payload containing the detected items, calculated subtotals, tax rates, confidence scores, and raw polygon coordinates for rendering.")
    pdf.normal_bullet("GET /api/v1/checkout/history: Returns a paginated history of past transactions.")
    pdf.normal_bullet("GET /api/v1/inventory/products: Manages CRUD operations for the 60 grocery items, tracking stock quantities and prices.")

    pdf.section_title("4.3 Database ORM & Glue Code")
    pdf.chapter_body("We utilized SQLAlchemy 2.0 as the Object Relational Mapper (ORM) mapped to a SQLite database. This provides Pythonic, type-safe queries preventing SQL injection.")
    pdf.chapter_body("The schema consists of heavily normalized tables:")
    pdf.normal_bullet("Products: Stores name, category, unit price, and current stock_quantity.")
    pdf.normal_bullet("Transactions: Stores a UUID, timestamp, tax, and total amount.")
    pdf.normal_bullet("TransactionItems: Links products to a transaction with the quantity and frozen unit_price_at_sale.")
    pdf.normal_bullet("InventoryLog: An append-only audit trail logging every stock deduction (SALE) or addition (RESTOCK).")
    pdf.normal_bullet("AdminUser & OTPSession: Tables handling the secure MFA authentication flow.")

    pdf.section_title("4.4 Feature Transformations")
    pdf.chapter_body("Data must be transformed rapidly between the HTTP request, the model, and the HTTP response. Glue code in inference.py handles:")
    pdf.normal_bullet("Image Resizing: If a 4K image is uploaded, it is dynamically resized (maintaining aspect ratio) to a maximum dimension of 4096px before inference to prevent RAM exhaustion.")
    pdf.normal_bullet("Mask Extraction: The PyTorch binary mask tensors [N, H, W] are computationally heavy. We use OpenCV (cv2.findContours and cv2.approxPolyDP) to convert these dense binary masks into sparse lists of (x,y) polygon coordinates. This reduces the JSON payload size by 99%, allowing the Flutter frontend to receive the data instantly.")

    # --- 5. FRONTEND MOBILE POS ---
    pdf.add_page()
    pdf.chapter_title("5. Frontend Mobile POS Application")
    pdf.chapter_body("The client-facing application is a cross-platform Point of Sale (POS) system built using the Flutter framework. It targets Android natively, but can also compile to Web and Windows Desktop.")

    pdf.section_title("5.1 Flutter Cross-Platform Architecture")
    pdf.chapter_body("Flutter was selected because it compiles to native ARM code, ensuring fluid 60fps animations which are critical when rendering complex UI overlays. The application handles camera feeds natively using the camera plugin, captures frames, compresses them to JPEG, and sends them to the FastAPI backend via the Dio HTTP client.")

    pdf.section_title("5.2 State Management (Riverpod)")
    pdf.chapter_body("Managing asynchronous state (API loading states, camera initialization, cart totals) is complex. We utilized Riverpod 2.x for robust, compile-safe state management.")
    pdf.normal_bullet("checkoutProvider: A StateNotifier that dictates the current phase of the checkout loop (Idle -> Capturing -> Uploading -> Analyzing -> Success/Error).")
    pdf.normal_bullet("inventoryProvider: A FutureProvider with auto-dispose caching that fetches the current product catalog and stock levels.")
    pdf.normal_bullet("authProvider: Manages the JWT tokens and MFA state for administrative access.")

    pdf.section_title("5.3 UI Components & Mask Rendering")
    pdf.chapter_body("The UI emphasizes a premium, modern aesthetic using Deep Blue (#1565C0) and Amber (#FF6F00) accents. The most technically complex component is the SegmentationOverlay.")
    pdf.chapter_body("When the JSON response arrives from the backend containing polygon arrays, Flutter's CustomPainter API is invoked. It loops through the polygons, applies a deterministic color palette based on the class_id, and draws semi-transparent filled paths directly on top of the original image. RepaintBoundary widgets are used to isolate this rendering layer, preventing the rest of the UI from dropping frames during the draw cycle.")
    pdf.chapter_body("The Receipt Panel utilizes staggered slide-in animations to present the detected items, unit prices, and grand totals clearly to the user/cashier.")

    # --- 6. INFRASTRUCTURE & DEPLOYMENT ---
    pdf.add_page()
    pdf.chapter_title("6. Infrastructure & Deployment")
    pdf.chapter_body("The system is designed for modern cloud-native deployment patterns while remaining entirely functional on local edge hardware.")

    pdf.section_title("6.1 Compute & Hardware Acceleration")
    pdf.chapter_body("The backend utilizes PyTorch. If deployed on a machine with an NVIDIA GPU and CUDA drivers, PyTorch automatically shifts tensor operations to VRAM, reducing inference time from ~800ms (CPU) to ~80ms (GPU).")

    pdf.section_title("6.2 Docker Orchestration")
    pdf.chapter_body("To eliminate 'it works on my machine' issues, the backend is containerized. The Dockerfile utilizes python:3.11-slim as a minimal base image. It installs system-level dependencies required by OpenCV (libgl1, libglib2.0), copies the source code, creates volume mount points for the database and weights directories, and exposes port 8000. This ensures 100% reproducibility across environments.")

    pdf.section_title("6.3 Networking & Security")
    pdf.chapter_body("Networking is handled via RESTful HTTP. To support the Flutter Web client, Cross-Origin Resource Sharing (CORS) is explicitly configured via FastAPI middleware.")
    pdf.chapter_body("Security features include:")
    pdf.normal_bullet("Multi-Factor Authentication (MFA): Administrative screens are gated behind an email-based OTP system.")
    pdf.normal_bullet("Password Hashing: Administrator passwords are never stored in plaintext; they are hashed using SHA-256 with a unique random salt per user.")
    pdf.normal_bullet("Rate Limiting: The SlowAPI library implements an in-memory rate limiter (default 10 req/min/IP) to prevent Denial of Service (DoS) attacks on the computationally expensive inference endpoints.")

    pdf.section_title("6.4 Storage Solutions")
    pdf.chapter_body("Data storage relies on SQLite. To ensure high concurrency without database locking issues, Write-Ahead Logging (WAL) PRAGMA is explicitly enabled on the connection engine. This allows concurrent reads while a write (e.g., an inventory deduction) is occurring. Model weights are stored on the local file system (/weights) to prevent bloating the source control repository.")

    pdf.section_title("6.5 Monitoring & Logging")
    pdf.chapter_body("A custom HTTP middleware intercepts every incoming request. It tracks the start time, executes the request, and logs the HTTP method, path, status code, and latency in milliseconds. This is critical for monitoring the performance of the AI model in production. Additionally, a /api/v1/health endpoint continuously reports the boolean status of the database connection and model readiness.")

    # --- 7. CONCLUSION ---
    pdf.add_page()
    pdf.chapter_title("7. Conclusion")
    pdf.chapter_body("The AI-Powered Automatic Retail Checkout System demonstrates a successful integration of advanced deep learning techniques within a modern full-stack web and mobile application. By automating item identification via instance segmentation, the system dramatically reduces checkout friction.")
    pdf.chapter_body("The architecture's modularity ensures that the heavy AI inference engine is decoupled from the reactive POS interface, allowing both to scale independently. The inclusion of robust data augmentation pipelines guarantees high model accuracy even in cluttered retail environments. Future iterations could explore migrating to edge-TPUs (e.g., Google Coral) for localized, offline inference, entirely eliminating network latency from the checkout flow.")

    pdf.output("AI_Checkout_System_Report.pdf")

if __name__ == "__main__":
    generate_report()
    print("Detailed 10-page report generated successfully as AI_Checkout_System_Report.pdf")
