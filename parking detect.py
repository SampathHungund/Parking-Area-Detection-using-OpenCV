import cv2
import numpy as np

input_type = "image"  
if input_type == "video":
    cap = cv2.VideoCapture("parking_lot_video.mp4")
else:
    img = cv2.imread("Upload a JPEG/PNG image along with the path")

lower_bound = np.array([180, 180, 180], dtype=np.uint8)
upper_bound = np.array([255, 255, 255], dtype=np.uint8)

while True:
    if input_type == "video":
        ret, frame = cap.read()
        if not ret:
            break
    else:
        frame = img.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    masked_frame = cv2.bitwise_and(thresh, mask)
    contours, _ = cv2.findContours(masked_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    empty_spots = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            empty_spots += 1

    cv2.putText(frame, f"Empty Spots: {empty_spots}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Parking Area Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if input_type == "video":
    cap.release()
cv2.destroyAllWindows()
