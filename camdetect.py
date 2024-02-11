import cv2
import torch

model_path = 'runs/train/exp5/weights/best.pt'  # model weights from training
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path) # set model
model.conf = 0.25 # adjust confidence threshold
# print(model)

cap = cv2.VideoCapture(0)  # 0 is default camera

while True:
    ret, frame = cap.read()  # read frame from the webcam
    if not ret:
        break

    results = model(frame) # inference

    results.render()  # bounding boxes and labels

    cv2.imshow('YOLOv5 Webcam Detection', frame)  # display

    if cv2.waitKey(1) == ord('q'):  # press 'q' to quit
        break

# exit
cap.release()
cv2.destroyAllWindows()