from ultralytics import YOLO
import cv2

count = 0
id_list = []
total_list = []

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    results1 = model.track(
        frame,
        line_width=1,
        persist=True,
        classes=0,
        conf=0.1,
        show=True,
        stream=True
    )

    for r1 in results1:
        boxes1 = r1.boxes
        masks1 = r1.masks
        probs1 = r1.probs

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
