import cv2
import torch

model_path = 'runs/train/exp5/weights/best.pt'  # model weights from training
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.conf = 0.25 # adjust confidence threshold

img_path = 'data/train/images/8-ounce-milk-carton-5_jpg.rf.5798c92c82665d39dbb0737c96cd1eb2.jpg' # image
img = cv2.imread(img_path)  # load image

results = model(img)

results.render()

cv2.imshow('YOLOv5 Detection', img) # display img

cv2.waitKey(0)  # Wait for any key to be pressed
cv2.destroyAllWindows()
