"""Generate a professionally formatted PDF viva study guide."""
from fpdf import FPDF

class VivaPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "AI-Powered Automatic Retail Checkout System - Viva Study Guide", align="C")
            self.ln(4)
            self.set_draw_color(13, 148, 136)
            self.set_line_width(0.3)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, num, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(13, 148, 136)
        self.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(13, 148, 136)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(15, 23, 42)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text, indent=15):
        x = self.get_x()
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(x + indent)
        self.cell(4, 5.5, "-")
        self.multi_cell(0, 5.5, text)
        self.ln(0.5)

    def bold_bullet(self, bold_part, rest, indent=15):
        x = self.get_x()
        self.set_x(x + indent)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(4, 5.5, "-")
        self.set_font("Helvetica", "B", 10)
        self.write(5.5, bold_part)
        self.set_font("Helvetica", "", 10)
        self.write(5.5, rest)
        self.ln(6)

    def qa_block(self, q_num, question, answer):
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(13, 148, 136)
        self.multi_cell(0, 5.5, f"Q{q_num}: {question}")
        self.ln(1)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(15)
        self.multi_cell(185, 5.5, answer)
        self.ln(2)

    def table_row(self, cols, widths, bold=False, header=False):
        h = 7
        self.set_font("Helvetica", "B" if bold or header else "", 9)
        if header:
            self.set_fill_color(13, 148, 136)
            self.set_text_color(255, 255, 255)
        else:
            self.set_fill_color(248, 250, 252)
            self.set_text_color(30, 30, 30)
        for i, col in enumerate(cols):
            self.cell(widths[i], h, col, border=1, fill=header, align="L" if i == 0 else "L")
        self.ln(h)

    def phase_block(self, title, bullets):
        self.ln(1)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(15, 23, 42)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        for b in bullets:
            self.bullet(b)
        self.ln(1)


def build_pdf():
    pdf = VivaPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ─── COVER PAGE ───
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(13, 148, 136)
    pdf.cell(0, 14, "Viva Study Guide", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "AI-Powered Automatic Retail", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, "Checkout System", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_draw_color(13, 148, 136)
    pdf.set_line_width(1)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Course: Deep Neural Networks", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Team: Varisha & Muzzammil", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, "Instance Segmentation | YOLOv8-seg | FastAPI | Flutter | MVTec D2S", align="C")

    # ─── 1. EXECUTIVE SUMMARY ───
    pdf.add_page()
    pdf.section_title("1", "Executive Summary")

    pdf.sub_title("The Problem")
    pdf.body_text("Traditional retail checkout is slow, error-prone, and labour-intensive. Cashiers must manually scan barcodes for every item, causing long queues, mis-scans, and high staffing costs.")

    pdf.sub_title("Our Solution")
    pdf.body_text("An end-to-end AI-powered checkout system that uses instance segmentation (YOLOv8-seg) to automatically identify grocery items placed on a counter from a single camera image. The system:")
    pdf.bullet("Detects and segments up to 60 grocery categories using the MVTec D2S dataset")
    pdf.bullet("Generates an itemised receipt with prices, quantities, and tax")
    pdf.bullet("Deducts inventory atomically and logs an audit trail")
    pdf.bullet("Presents everything through a Flutter mobile POS interface with real-time segmentation overlays")
    pdf.bullet("Secures admin access with SHA-256 + MFA (email OTP + JWT)")

    pdf.sub_title("Why Instance Segmentation?")
    pdf.bullet("Items on a checkout counter overlap - segmentation masks handle occlusion precisely")
    pdf.bullet("Pixel-level masks give accurate visual feedback to the cashier")
    pdf.bullet("Polygon coordinates are returned to the frontend for rendering overlays")

    # ─── 2. END-TO-END WORKFLOW ───
    pdf.add_page()
    pdf.section_title("2", "End-to-End Workflow")

    pdf.phase_block("Phase 1 - Requirements & Dataset Selection", [
        "Identified the need for a DNN-based cashier-less checkout system",
        "Selected MVTec D2S (Densely Segmented Supermarket) - 60 grocery categories with COCO-format polygon annotations",
        "Training split: single-item images (clean backgrounds); Validation: multi-item scenes with occlusion",
    ])

    pdf.phase_block("Phase 2 - Data Preparation", [
        "Wrote data_preparation.py to convert COCO JSON to YOLO segmentation format",
        "Each label file: class_id x1 y1 x2 y2 ... with coordinates normalised to [0,1]",
        "Generated d2s.yaml config mapping 60 class IDs (0-59) to human-readable names",
        "Skipped RLE-encoded masks and degenerate polygons (<3 points)",
    ])

    pdf.phase_block("Phase 3 - Model Training", [
        "Primary model: YOLOv8n-seg (nano) with COCO-pretrained transfer learning",
        "Optimizer: SGD (lr=0.01, momentum=0.937, cosine LR schedule)",
        "Augmentations: mosaic, rotation +/-15 deg, HSV jitter, flips, scale 0.5-1.5x",
        "Early stopping (patience=20), checkpoints every 10 epochs",
        "Alternative: Mask R-CNN (ResNet-50-FPN via Detectron2) - switchable via env var",
        "Training on Google Colab GPU (train_colab.ipynb)",
    ])

    pdf.phase_block("Phase 4 - Validation & Evaluation", [
        "COCO metrics: mAP@50, mAP@50-95, mAP@75, per-class AP for all 60 classes",
        "validate.py generates full evaluation reports with confusion matrix and PR curves",
    ])

    pdf.phase_block("Phase 5 - Backend Development (FastAPI)", [
        "RESTful API with 8 endpoints (scan, history, inventory CRUD, health, auth)",
        "Singleton InferenceEngine loads model once at startup (zero per-request overhead)",
        "Thread-safe inference via asyncio lock for concurrent requests",
        "SQLite + SQLAlchemy ORM with 6 tables; Pydantic v2 for validation",
        "Rate limiting (SlowAPI), CORS middleware, structured JSON error handling",
    ])

    pdf.phase_block("Phase 6 - Frontend Development (Flutter)", [
        "Cross-platform POS (Android, iOS, Web) with Riverpod 2.x state management",
        "6 screens: Login, OTP, Checkout, Receipt, Inventory, History",
        "SegmentationOverlay widget draws masks at 60fps via CustomPainter",
        "Dio HTTP client with retry logic and exponential backoff",
        "Premium design: teal-coral palette, Google Fonts (Inter + Outfit)",
    ])

    pdf.phase_block("Phase 7 - Security & Authentication", [
        "Two-step MFA: password (SHA-256+salt) then email OTP then JWT token",
        "Account lockout after 5 failed attempts (15-minute cooldown)",
        "OTP brute-force protection (max 5 attempts per session, 5-min expiry)",
        "Auth gate wraps entire app - unauthenticated users see only login screen",
    ])

    pdf.phase_block("Phase 8 - Mobile Deployment", [
        "Built release APK for physical Android devices",
        "Platform-aware backend URL (emulator vs physical device vs web)",
        "Native camera integration + file import for image analysis",
    ])

    # ─── 3. TECHNICAL ARCHITECTURE ───
    pdf.add_page()
    pdf.section_title("3", "Technical Architecture & Decisions")

    pdf.sub_title("3.1 System Architecture")
    pdf.body_text("Camera/Image -> Flutter App -> FastAPI Backend -> YOLOv8-seg -> Detections")
    pdf.body_text("FastAPI also connects to SQLite Database (Products, Transactions, Inventory) and returns JSON Response with Masks, Receipt, and Totals back to the Flutter UI.")

    pdf.sub_title("3.2 Tech Stack Justification")
    w = [30, 40, 120]
    pdf.table_row(["Layer", "Technology", "Why This Over Alternatives"], w, header=True)
    pdf.table_row(["DL Model", "YOLOv8-seg", "Best speed-accuracy trade-off; single-stage; fast inference"], w)
    pdf.table_row(["Alt. Model", "Mask R-CNN", "Higher accuracy on small objects; two-stage; included for comparison"], w)
    pdf.table_row(["Backend", "FastAPI", "Async support; auto Swagger docs; Pydantic integration; faster than Flask"], w)
    pdf.table_row(["Database", "SQLite+ORM", "Zero config; file-based; SQLAlchemy for type-safe queries"], w)
    pdf.table_row(["Frontend", "Flutter", "Single codebase for Android/iOS/Web; 60fps rendering; Riverpod state"], w)
    pdf.table_row(["HTTP", "Dio", "Interceptors; retry with backoff; multipart uploads"], w)
    pdf.table_row(["Auth", "SHA-256+JWT", "Industry-standard MFA; stateless tokens; no external service needed"], w)

    pdf.sub_title("3.3 Key Design Patterns")
    pdf.bold_bullet("Singleton Pattern ", "- InferenceEngine loads model once, shared across requests")
    pdf.bold_bullet("Repository Pattern ", "- SQLAlchemy ORM abstracts database operations")
    pdf.bold_bullet("Provider Pattern ", "- Riverpod providers manage UI state reactively")
    pdf.bold_bullet("Atomic Transactions ", "- Inventory deduction is all-or-nothing per checkout")
    pdf.bold_bullet("Middleware Pipeline ", "- Request logging, CORS, rate limiting as composable layers")

    pdf.sub_title("3.4 Database Schema (6 Tables)")
    w2 = [45, 145]
    pdf.table_row(["Table", "Purpose"], w2, header=True)
    pdf.table_row(["products", "60 D2S categories with name, price, stock, barcode"], w2)
    pdf.table_row(["transactions", "Checkout records with UUID, totals, timestamps"], w2)
    pdf.table_row(["transaction_items", "Line items per transaction (FK to product)"], w2)
    pdf.table_row(["inventory_log", "Audit trail: SALE/RESTOCK with quantity deltas"], w2)
    pdf.table_row(["admin_users", "Admin accounts with hashed passwords, lockout tracking"], w2)
    pdf.table_row(["otp_sessions", "Time-limited OTP codes for MFA"], w2)

    pdf.sub_title("3.5 Scan Endpoint Request Flow (10 Steps)")
    steps = [
        "1. Flutter sends multipart image via POST /api/v1/checkout/scan",
        "2. FastAPI validates MIME type and size (max 20MB)",
        "3. Image decoded with OpenCV, resized if >4096px",
        "4. Singleton InferenceEngine runs YOLOv8-seg inference",
        "5. Detections: per-instance class, confidence, bbox, mask polygon",
        "6. item_counter aggregates detections by class name",
        "7. Price lookup in SQLite, build receipt line items",
        "8. Atomically deduct inventory, write audit log",
        "9. Persist transaction record with UUID",
        "10. Return JSON with items, prices, masks, and totals",
    ]
    for s in steps:
        pdf.bullet(s)

    # ─── 4. VIVA Q&A ───
    pdf.add_page()
    pdf.section_title("4", "Anticipated Viva Questions & Answers")

    # --- A. Basic ---
    pdf.sub_title("A. Basic Concept Questions")

    pdf.qa_block(1,
        "What is instance segmentation and how does it differ from object detection and semantic segmentation?",
        "Object detection draws bounding boxes around objects. Semantic segmentation labels every pixel with a class but doesn't distinguish instances. Instance segmentation does both - it identifies each individual object AND provides a pixel-level mask. For our checkout, this means we can separately identify two bananas even if they overlap, and draw precise outlines around each."
    )
    pdf.qa_block(2,
        "Why did you choose YOLOv8-seg over other models?",
        "YOLOv8-seg is a single-stage detector that performs detection and segmentation in one forward pass, making it very fast (suitable for real-time checkout). It has excellent pretrained COCO weights for transfer learning, a clean Python API via Ultralytics, and achieves competitive accuracy. Compared to Mask R-CNN (two-stage), YOLOv8 is 3-5x faster at inference."
    )
    pdf.qa_block(3,
        "What is the MVTec D2S dataset?",
        "MVTec D2S (Densely Segmented Supermarket) is a benchmark dataset for grocery product recognition. It contains 60 grocery categories with high-resolution images and COCO-format polygon annotations. Training images show single items on clean backgrounds; validation images show multi-item scenes with occlusion - simulating real checkout scenarios."
    )
    pdf.qa_block(4,
        "What is transfer learning and why did you use it?",
        "Transfer learning means starting with a model pretrained on a large dataset (COCO, 80 classes, millions of images) and fine-tuning it on our smaller D2S dataset (60 classes). The pretrained model already understands general visual features (edges, textures, shapes). We only need to teach it our specific grocery categories, requiring less data and training time."
    )
    pdf.qa_block(5,
        "What is COCO format and why convert to YOLO format?",
        "COCO stores annotations as JSON with polygon coordinates in absolute pixel values. YOLOv8 requires a specific text format: one .txt file per image with 'class_id x1 y1 x2 y2 ...' in normalised (0-1) coordinates. Our data_preparation.py handles this conversion, including category ID remapping (COCO 1-60 to YOLO 0-59)."
    )
    pdf.qa_block(6,
        "What is Non-Maximum Suppression (NMS)?",
        "NMS removes duplicate detections. When the model predicts multiple overlapping bounding boxes for the same object, NMS keeps only the one with highest confidence and suppresses others with IoU above a threshold (we use 0.45). This prevents counting one item multiple times."
    )
    pdf.qa_block(7,
        "Explain the role of the confidence threshold.",
        "We set a confidence threshold of 0.35 - detections below this score are discarded. Too low means false positives (phantom items on the receipt); too high means missed items (customer charged less). 0.35 was tuned to balance precision and recall for our grocery categories."
    )

    # --- B. Implementation ---
    pdf.sub_title("B. Implementation & Workflow Questions")

    pdf.qa_block(8,
        "Walk us through what happens when a user scans an image.",
        "1) Flutter captures/imports an image and sends it as multipart/form-data to /api/v1/checkout/scan. 2) FastAPI validates file type and size. 3) Image is decoded with OpenCV and resized if needed. 4) Singleton InferenceEngine runs YOLOv8-seg. 5) Detections are aggregated by class. 6) Each class is matched to a product in SQLite. 7) Stock is atomically deducted with audit log. 8) Transaction is persisted. 9) JSON response with items, prices, masks, and totals is returned. 10) Flutter renders segmentation overlays and receipt panel."
    )
    pdf.qa_block(9,
        "Why the Singleton pattern for InferenceEngine?",
        "Loading a deep learning model is expensive (takes seconds, uses hundreds of MB of RAM/VRAM). The Singleton ensures the model loads exactly once at startup and is shared across all requests, giving zero-overhead inference per request. The asyncio lock ensures thread-safety for concurrent requests."
    )
    pdf.qa_block(10,
        "How does your authentication system work?",
        "Two-factor authentication: Step 1 - User submits username + password, verified against SHA-256(salt + password) hash. If valid, a 6-digit OTP is generated, hashed, and emailed. Step 2 - User submits OTP + session token, hash is verified, and a JWT access token is issued. Security includes account lockout after 5 failures, OTP expiry (5 min), brute-force protection, and rate limiting."
    )
    pdf.qa_block(11,
        "How does inventory management work?",
        "Inventory deduction is atomic - if any part fails, the entire operation rolls back. Stock is clamped at zero. Every change (SALE or RESTOCK) creates an inventory_log entry for auditing. Out-of-stock items are reported in the items_not_found field."
    )
    pdf.qa_block(12,
        "Explain your Flutter state management.",
        "We use Riverpod 2.x with three providers: authProvider (StateNotifier for auth flow), checkoutProvider (scan state: idle/loading/response/error), and inventoryProvider (FutureProvider with auto-dispose caching). Riverpod was chosen over Provider/Bloc because it's compile-safe, testable, and supports auto-disposal."
    )
    pdf.qa_block(13,
        "How do you render segmentation masks on the UI?",
        "The SegmentationOverlay widget uses Flutter's CustomPainter with RepaintBoundary for 60fps rendering. It receives polygon coordinates from the API, scales them to displayed image dimensions, and draws filled semi-transparent polygons with a 20-colour palette cycled by class_id."
    )
    pdf.qa_block(14,
        "What data augmentations did you apply and why?",
        "Mosaic (100%): combines 4 images to simulate dense scenes. Flips (50%): products appear in any orientation. Rotation (+/-15 deg): items aren't always aligned. Scale (0.5-1.5x): products vary in size. HSV jitter: handles lighting variations. These improve generalisation to real-world scenarios."
    )

    # --- C. Challenges ---
    pdf.sub_title("C. Challenge & Troubleshooting Questions")

    pdf.qa_block(15,
        "What was the biggest challenge?",
        "Dataset conversion. MVTec D2S uses a non-standard directory structure (flat image folder, D2S-specific JSON naming). Our initial converter assumed standard COCO layout. We rewrote data_preparation.py to handle the actual structure, try multiple image paths, and correctly remap category IDs from COCO's 1-indexed to YOLO's 0-indexed."
    )
    pdf.qa_block(16,
        "How do you handle items detected but not in the database?",
        "A 4-strategy matching pipeline: 1) Exact name match, 2) Case-insensitive partial match, 3) D2S-to-product name mapping from seed data, 4) Barcode field substring match. If all fail, the item goes into items_not_found and is reported to the cashier with no charge."
    )
    pdf.qa_block(17,
        "How do you handle concurrent scan requests?",
        "The InferenceEngine uses asyncio.Lock() to serialise inference calls, preventing race conditions on the GPU. Database operations use SQLAlchemy's with_for_update() for row-level locking during stock deduction."
    )
    pdf.qa_block(18,
        "What happens if the model isn't loaded?",
        "The health endpoint reports 'degraded' status. The scan endpoint returns HTTP 503 (Service Unavailable). The frontend shows an error message. Inventory and history endpoints still work - graceful degradation."
    )
    pdf.qa_block(19,
        "How did you handle web-to-mobile transition?",
        "Conditional imports for platform-specific camera (live_camera_web.dart vs live_camera_stub.dart). Auto-detection of platform for backend URL. Added file_picker for device storage import. Configured Android manifest for camera/internet/storage permissions."
    )
    pdf.qa_block(20,
        "Why SQLite instead of PostgreSQL?",
        "SQLite is embedded, zero-configuration, file-based - perfect for a single-store POS demo. For production multi-store deployment, we'd migrate to PostgreSQL. SQLAlchemy's ORM abstraction makes this trivial - only the connection string changes."
    )
    pdf.qa_block(21,
        "What evaluation metrics did you use?",
        "mAP@50: masks overlap ground truth by >=50%. mAP@50-95: average over stricter IoU thresholds. Per-class AP: identifies strong vs weak categories. We evaluate both box (detection) and mask (segmentation) metrics separately."
    )
    pdf.qa_block(22,
        "What are the system's limitations?",
        "Limited to 60 D2S categories. Single-camera static images (no video yet). SQLite limits concurrent writes. No barcode fallback. Requires good lighting for reliable detection."
    )
    pdf.qa_block(23,
        "How would you improve for production?",
        "Add real-time video processing. Migrate to PostgreSQL. Implement barcode scanner fallback. Add product registration pipeline for new items. Deploy with TensorRT/ONNX for faster inference. Add Docker and CI/CD."
    )
    pdf.qa_block(24,
        "What is the d2s.yaml file?",
        "The Ultralytics dataset config that tells YOLOv8 where to find train/val images, the number of classes (60), and class index-to-name mapping. Auto-generated by data_preparation.py, referenced by train.py and validate.py."
    )
    pdf.qa_block(25,
        "How does the system handle overlapping items?",
        "This is why we chose instance segmentation. YOLOv8-seg produces separate masks per instance even when overlapping. The segmentation head predicts per-pixel masks for each bounding box. Mosaic augmentation during training specifically simulates dense scenes to improve separation."
    )

    # ─── QUICK REFERENCE ───
    pdf.add_page()
    pdf.section_title("", "Quick Reference Card")
    w3 = [50, 140]
    pdf.table_row(["Aspect", "Detail"], w3, header=True)
    pdf.table_row(["Model", "YOLOv8n-seg (nano), 640x640 input"], w3)
    pdf.table_row(["Dataset", "MVTec D2S, 60 classes, COCO JSON"], w3)
    pdf.table_row(["Backend", "FastAPI + SQLite + SQLAlchemy"], w3)
    pdf.table_row(["Frontend", "Flutter 3.16+ + Riverpod 2.x"], w3)
    pdf.table_row(["Auth", "SHA-256 + Email OTP + JWT"], w3)
    pdf.table_row(["Key Metrics", "mAP@50, mAP@50-95, per-class AP"], w3)
    pdf.table_row(["Augmentations", "Mosaic, flip, rotate, scale, HSV jitter"], w3)
    pdf.table_row(["Optimizer", "SGD, lr=0.01, cosine schedule"], w3)
    pdf.table_row(["Early Stopping", "Patience = 20 epochs"], w3)
    pdf.table_row(["Tax Rate", "10% on all transactions"], w3)
    pdf.table_row(["Confidence", "0.35 threshold, NMS IoU=0.45"], w3)
    pdf.table_row(["DB Tables", "products, transactions, items, inventory_log, admin, otp"], w3)

    # Save
    output = r"C:\Users\Muzzammil Idrees\OneDrive\Desktop\Viva_Study_Guide.pdf"
    pdf.output(output)
    print(f"PDF saved to: {output}")

if __name__ == "__main__":
    build_pdf()
