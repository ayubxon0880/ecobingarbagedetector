from ultralytics import YOLO
import cv2

count = 0
id_list = []
total_list = []
model = YOLO('best.pt')

results1 = model.track(
        "Video-2 (1).mp4",
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
            
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
