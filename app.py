import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import time
from supabase import create_client
import uuid
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS untuk dashboard

# Initialize Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

def process_image(image_bytes):
    """
    Process image menggunakan HSV color segmentation + morphology + contour
    ⚠️ TIDAK DIUBAH - SAMA PERSIS dengan algoritma original
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
        max_width = 640  # Resize ke max 640px width
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
    """
    Klasifikasi kendaraan berdasarkan rasio h/w
    ⚠️ TIDAK DIUBAH - SAMA PERSIS dengan algoritma original
    """
    if ratio_hw is None:
        return "tidak terdeteksi", 0.0
    
    # Sesuai tabel klasifikasi
    if 0.6 <= ratio_hw < 1.0:
        return "mobil", 17
    elif 1.0 <= ratio_hw < 1.2:
        return "truk/bus sedang", 20
    elif 1.2 <= ratio_hw <= 1.5:
        return "truk/bus besar", 34
    else:
        # Di luar rentang (ratio < 0.6 atau > 1.5)
        return "kendaraan tidak terdefinisi", 0.0

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

def insert_to_database(log_data, image_data):
    """
    ✅ NEW FUNCTION: Insert detection ke database (2 tabel)
    """
    if not supabase:
        print("Supabase not configured - skipping database insert")
        return False
    
    try:
        # 1. INSERT ke overtaking_logs
        log_response = supabase.table("overtaking_logs").insert(log_data).execute()
        
        if not log_response.data:
            print("Failed to insert into overtaking_logs")
            return False
        
        print(f"✅ Inserted to overtaking_logs: {log_data['id']}")
        
        # 2. INSERT ke overtaking_images
        image_response = supabase.table("overtaking_images").insert(image_data).execute()
        
        if not image_response.data:
            print("Failed to insert into overtaking_images")
            return False
        
        print(f"✅ Inserted to overtaking_images: {image_data['overtaking_log_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error inserting to database: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route("/api/classify-image", methods=["POST"])
def classify_image():
    """
    ✅ UPDATED ENDPOINT: /classify → /api/classify-image
    Menerima gambar, proses, simpan ke database, return minimal response
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
        image_size_kb = len(image_bytes) / 1024
        if len(image_bytes) > 5 * 1024 * 1024:
            return jsonify({
                "success": False,
                "error": "File too large. Max 5MB"
            }), 413
        
        print(f"Received image: {image_size_kb:.2f} KB")
        
        # ============================
        # IMAGE PROCESSING (TIDAK DIUBAH)
        # ============================
        ratio_hw, processed_img = process_image(image_bytes)
        
        if ratio_hw is None:
            return jsonify({
                "success": False,
                "error": "Vehicle not detected in image"
            }), 400
        
        # Klasifikasi (TIDAK DIUBAH)
        vehicle_type, detected_length_m = classify_vehicle(ratio_hw)
        
        # Upload ke Supabase Storage
        image_url = upload_to_supabase(image_bytes)
        
        if not image_url:
            return jsonify({
                "success": False,
                "error": "Failed to upload image to storage"
            }), 500
        
        # Hitung waktu eksekusi
        classification_time = round(time.time() - start_time, 3)
        
        # ============================
        # NEW: Generate overtaking_log_id & INSERT ke database
        # ============================
        overtaking_log_id = str(uuid.uuid4())
        
        # Data untuk overtaking_logs
        log_data = {
            "id": overtaking_log_id,
            "vehicle_type": vehicle_type,
            "detected_length_m": detected_length_m,
            "ratio_hw": round(ratio_hw, 3),
            "classification_time": classification_time,
            "vehicle_speed": None,
            "distance_ab": None,
            "feasibility_result": None,
            "feasibility_time": None,
            "total_process_time": None
        }
        
        # Data untuk overtaking_images
        image_data = {
            "id": str(uuid.uuid4()),
            "overtaking_log_id": overtaking_log_id,
            "image_url": image_url,
            "image_size_kb": round(image_size_kb, 2)
        }
        
        # Insert ke database
        db_success = insert_to_database(log_data, image_data)
        
        if not db_success:
            print("⚠️ Warning: Failed to insert to database, but continuing...")
        
        # ============================
        # RESPONSE MINIMAL KE ESP
        # ============================
        response_data = {
            "success": True,
            "overtaking_log_id": overtaking_log_id,
            "vehicle_type": vehicle_type,
            "detected_length_m": detected_length_m
        }
        
        print(f"✅ Classification done: {vehicle_type} ({detected_length_m}m, ratio: {ratio_hw:.2f}) in {classification_time}s")
        
        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/update-stm", methods=["POST"])
def update_stm_result():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data received"
            }), 400
        
        overtaking_log_id = data.get("overtaking_log_id")
        if not overtaking_log_id:
            return jsonify({
                "success": False,
                "error": "Missing overtaking_log_id"
            }), 400
        
        update_data = {
            "vehicle_speed": data.get("vehicle_speed"),
            "distance_ab": data.get("distance_ab"),
            "feasibility_result": data.get("feasibility_result"),
            "feasibility_time": data.get("feasibility_time"),
            "total_process_time": data.get("total_process_time")
        }
        
        if all(v is None for v in update_data.values()):
            return jsonify({
                "success": False,
                "error": "No data to update"
            }), 400
        
        # ✅ DEBUG: Print semua info
        print("=" * 50)
        print(f"📝 Received overtaking_log_id: '{overtaking_log_id}'")
        print(f"📝 Type: {type(overtaking_log_id)}")
        print(f"📝 Length: {len(overtaking_log_id)}")
        print(f"📝 Update data: {update_data}")
        print("=" * 50)
        
        if not supabase:
            return jsonify({
                "success": False,
                "error": "Supabase not configured"
            }), 500
        
        # ✅ DEBUG: Cek dulu apakah row ada
        check_response = supabase.table("overtaking_logs").select("id").eq("id", overtaking_log_id).execute()
        print(f"🔍 Check query result: {check_response.data}")
        
        if not check_response.data or len(check_response.data) == 0:
            print(f"❌ Log ID NOT FOUND in database: {overtaking_log_id}")
            return jsonify({
                "success": False,
                "error": f"Log ID not found: {overtaking_log_id}"
            }), 404
        
        print(f"✅ Log ID FOUND in database!")
        
        # UPDATE database
        response = supabase.table("overtaking_logs")\
            .update(update_data)\
            .eq("id", overtaking_log_id)\
            .execute()
        
        print(f"🔄 Update response: {response.data}")
        
        if not response.data or len(response.data) == 0:
            return jsonify({
                "success": False,
                "error": f"Update failed for log ID: {overtaking_log_id}"
            }), 500
        
        print(f"✅ STM result updated for log: {overtaking_log_id}")
        
        return jsonify({
            "success": True,
            "message": "STM result updated successfully",
            "updated_log_id": overtaking_log_id
        }), 200
        
    except Exception as e:
        print(f"❌ Error updating STM result: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """
    ✅ NEW ENDPOINT: Stats untuk dashboard (optional tapi berguna)
    """
    try:
        if not supabase:
            return jsonify({
                "success": False,
                "error": "Supabase not configured"
            }), 500
        
        # Total detections
        total_response = supabase.table("overtaking_logs").select("id", count="exact").execute()
        total_count = total_response.count if total_response.count else 0
        
        # Today's detections
        today = datetime.now().strftime("%Y-%m-%d")
        today_response = supabase.table("overtaking_logs").select("id", count="exact").gte("created_at", today).execute()
        today_count = today_response.count if today_response.count else 0
        
        # Vehicle type distribution
        all_logs = supabase.table("overtaking_logs").select("vehicle_type").execute()
        vehicle_distribution = {}
        if all_logs.data:
            for log in all_logs.data:
                vtype = log.get("vehicle_type", "unknown")
                vehicle_distribution[vtype] = vehicle_distribution.get(vtype, 0) + 1
        
        # Feasibility distribution (yang sudah diupdate STM)
        feasibility_logs = supabase.table("overtaking_logs").select("feasibility_result").not_.is_("feasibility_result", "null").execute()
        feasibility_distribution = {}
        if feasibility_logs.data:
            for log in feasibility_logs.data:
                fresult = log.get("feasibility_result", "unknown")
                feasibility_distribution[fresult] = feasibility_distribution.get(fresult, 0) + 1
        
        return jsonify({
            "success": True,
            "total_detections": total_count,
            "today_detections": today_count,
            "vehicle_distribution": vehicle_distribution,
            "feasibility_distribution": feasibility_distribution
        }), 200
        
    except Exception as e:
        print(f"❌ Error getting stats: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint untuk monitoring"""
    return jsonify({
        "status": "healthy",
        "message": "Classification API is running",
        "supabase_configured": supabase is not None,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route("/", methods=["GET"])
def home():
    """Root endpoint untuk keep-alive ping"""
    return jsonify({
        "service": "Vehicle Classification API",
        "version": "3.0",
        "endpoints": {
            "/api/classify-image": "POST - Upload image for classification (ESP → Render)",
            "/api/update-stm": "POST - Update STM result (ESP → Render)",
            "/api/stats": "GET - Get statistics (Dashboard)",
            "/health": "GET - Health check"
        }
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print(f"🚀 Starting Vehicle Classification API v3.0")
    print(f"📡 Port: {port}")
    print(f"🗄️  Supabase configured: {supabase is not None}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)