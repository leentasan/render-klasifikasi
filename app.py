import os
from flask import Flask, request, jsonify
import numpy as np
import cv2
import time
from supabase import create_client
import uuid
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

def process_image(image_bytes):
    """
    Process image menggunakan HSV color segmentation + morphology + contour
    SAMA PERSIS dengan algoritma Colab notebook
    """
    try:
        # Decode image dari bytes
        nparr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Gambar tidak valid atau tidak terbaca.")
        
        # ============================
        # Resize jika terlalu besar
        # ============================
        h, w = img.shape[:2]
        max_width = 800  # Resize ke max 800px width
        if w > max_width:
            ratio = max_width / w
            new_w = max_width
            new_h = int(h * ratio)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Image resized from {w}x{h} to {new_w}x{new_h}")
        # ============================
        
        h, w = img.shape[:2]  # Update h, w setelah resize
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ============================
        # Auto color segmentation (patch tengah)
        # ============================
        center_ratio = 0.3
        h_tol, s_tol, v_tol = 15, 60, 60

        ch, cw = int(h*center_ratio), int(w*center_ratio)
        y1, y2 = h//2 - ch//2, h//2 + ch//2
        x1, x2 = w//2 - cw//2, w//2 + cw//2
        patch = hsv[y1:y2, x1:x2]

        mask_valid = (patch[:,:,1] > 40) & (patch[:,:,2] > 40)
        sel = patch[mask_valid].reshape(-1,3) if np.count_nonzero(mask_valid) > 50 else patch.reshape(-1,3)
        h0, s0, v0 = sel.mean(axis=0).astype(int)

        lower = np.array([max(0, h0-h_tol), max(0, s0-s_tol), max(0, v0-v_tol)])
        upper = np.array([min(179, h0+h_tol), min(255, s0+s_tol), min(255, v0+v_tol)])

        mask_raw = cv2.inRange(hsv, lower, upper)

        # ============================
        # Morphology untuk merapikan mask
        # ============================
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
        mask = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # ============================
        # Ambil KONTUR LANGSUNG dari mask
        # ============================
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("Peringatan: Tidak ada kontur ditemukan.")
            return None, None

        # Pilih kontur terbesar
        c = max(contours, key=cv2.contourArea)
        
        # Haluskan dengan approx
        epsilon = 0.005 * cv2.arcLength(c, True)
        c_approx = cv2.approxPolyDP(c, epsilon, True)
        
        # Bounding box
        x, y, bw, bh = cv2.boundingRect(c_approx)
        ratio_hw = bh / float(bw) if bw > 0 else 0

        print(f"Bounding box (x,y,w,h): ({x},{y},{bw},{bh}), Ratio h/w: {ratio_hw:.2f}")
        
        return ratio_hw, img

    except Exception as e:
        print(f"Error dalam process_image: {e}")
        return None, None

def classify_vehicle(ratio_hw):
    """Klasifikasi kendaraan berdasarkan rasio h/w"""
    if ratio_hw is None:
        return "tidak terdeteksi", 0.0
    
    # Sesuaikan dengan threshold kamu
    if ratio_hw < 0.6:
        return "mobil", 4.5
    elif 0.6 <= ratio_hw < 1.0:
        return "bus sedang", 8.0
    elif 1.0 <= ratio_hw < 1.3:
        return "bus besar", 12.0
    else:
        return "truk besar", 15.0

def upload_to_supabase(image_bytes):
    """Upload gambar ke Supabase Storage"""
    if not supabase:
        print("Supabase not configured - skipping upload")
        return None
    
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_path = f"{file_id}.jpg"
        
        # Upload to Supabase Storage bucket "overtaking-images"
        res = supabase.storage.from_("overtaking-images").upload(
            file_path,
            image_bytes,
            {"content-type": "image/jpeg"}
        )
        
        # Get public URL
        image_url = supabase.storage.from_("overtaking-images").get_public_url(file_path)
        
        print(f"Image uploaded to Supabase: {image_url}")
        return image_url
        
    except Exception as e:
        print(f"Error uploading to Supabase: {e}")
        return None

@app.route("/classify", methods=["POST"])
def classify_image():
    """
    Endpoint untuk klasifikasi gambar dari ESP32
    Menerima multipart/form-data dengan field 'image'
    """
    start_time = time.time()
    
    # Validasi upload
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded. Use 'image' field in multipart/form-data"
        }), 400

    try:
        # Baca image dari request
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Validasi ukuran (max 5MB)
        if len(image_bytes) > 5 * 1024 * 1024:
            return jsonify({
                "success": False,
                "error": "File too large. Max 5MB"
            }), 413
        
        print(f"Received image: {len(image_bytes)} bytes")
        
        # Process image (algoritma SAMA)
        ratio_hw, processed_img = process_image(image_bytes)
        
        # Klasifikasi
        vehicle_type, detected_length_m = classify_vehicle(ratio_hw)
        
        # Upload ke Supabase Storage
        image_url = upload_to_supabase(image_bytes)
        
        # Hitung waktu eksekusi
        classification_time = round(time.time() - start_time, 2)
        
        # Response untuk ESP
        response_data = {
            "success": True,
            "vehicle_type": vehicle_type,
            "detected_length_m": detected_length_m,
            "image_url": image_url,
            "ratio_hw": round(ratio_hw, 2) if ratio_hw else None,
            "classification_time": classification_time,
            "status": "success"
        }
        
        print(f"Classification done: {vehicle_type} (ratio: {ratio_hw:.2f}) in {classification_time}s")
        
        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "failed"
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint untuk monitoring"""
    return jsonify({
        "status": "healthy",
        "message": "Classification API is running",
        "supabase_configured": supabase is not None
    }), 200

@app.route("/", methods=["GET"])
def home():
    """Root endpoint untuk keep-alive ping"""
    return jsonify({
        "service": "Vehicle Classification API",
        "version": "2.0",
        "endpoints": {
            "/classify": "POST - Upload image for classification",
            "/health": "GET - Health check"
        }
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}")
    print(f"Supabase configured: {supabase is not None}")
    app.run(host="0.0.0.0", port=port, debug=False)