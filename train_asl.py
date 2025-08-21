from ultralytics import YOLO
import cv2
import os

# ======================
# CONFIGURATION
# ======================
YOLO_WEIGHTS = "runs/train/ASL_YOLO/weights/best.pt"  # trained model
TEST_IMAGE = "data/test/images/A22_jpg.rf.92993c024714ac07ce898e3da87fe94e.jpg"
SAVE_DIR = "inference_results"

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# LOAD MODEL
# ======================
model = YOLO(YOLO_WEIGHTS)
print(f"Loaded model from {YOLO_WEIGHTS}")

# ======================
# INFERENCE ON IMAGE
# ======================
if os.path.exists(TEST_IMAGE):
    results = model.predict(source=TEST_IMAGE)

    # Show annotated image
    results[0].show()

    # Save annotated image correctly
    results[0].save()  # saves in default "runs/predict" folder
else:
    print(f"Test image not found at {TEST_IMAGE}")

# ======================
# REAL-TIME WEBCAM INFERENCE
# ======================
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 'q' to quit webcam inference.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()  # returns numpy array

    # Show frame
    cv2.imshow("ASL YOLO Webcam", annotated_frame)

    cv2.imwrite(os.path.join(SAVE_DIR, "frame.jpg"), annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
