from flask import Flask, request, jsonify
import numpy as np
import cv2
import time
from sklearn.cluster import KMeans
import paho.mqtt.publish as publish

app = Flask(__name__)

MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC = "overtaking/classification"

@app.route("/classify", methods=["POST"])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_file = request.files['image'].read()
        image_array = np.asarray(bytearray(image_file), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        # Ekstrak pusat gambar
        center = image_rgb[height//3:2*height//3, width//3:2*width//3]
        center_pixels = center.reshape(-1, 3)
        all_pixels = image_rgb.reshape(-1, 3)
        sampled_pixels = np.vstack((all_pixels[::30], center_pixels))

        kmeans = KMeans(n_clusters=2, random_state=42)
        start_time = time.time()
        kmeans.fit(sampled_pixels)

        labels = kmeans.predict(all_pixels).reshape(height, width)
        center_labels = labels[height//3:2*height//3, width//3:2*width//3]
        object_cluster = np.argmax(np.bincount(center_labels.flatten()))
        mask = np.uint8((labels == object_cluster) * 255)

        # Kontur
        kernel = np.ones((5, 5), np.uint8)
        processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vehicle_type = "tidak terdeteksi"
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            ratio = round(w / h, 2)
            print(f"Bounding box rasio: {ratio}")

            if 0.6 <= ratio < 1.0:
                vehicle_type = "mobil"
            elif 1.0 <= ratio < 1.3:
                vehicle_type = "bus sedang"
            elif 1.3 <= ratio <= 1.6:
                vehicle_type = "bus besar"

        classification_time = round(time.time() - start_time, 2)

        # Kirim ke MQTT
        message = {
            "vehicle_type": vehicle_type,
            "classification_time": classification_time
        }
        publish.single(MQTT_TOPIC, payload=str(message), hostname=MQTT_BROKER)

        return jsonify({
            "vehicle_type": vehicle_type,
            "classification_time": classification_time
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
