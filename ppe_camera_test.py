import cv2
from ultralytics import YOLO


model = YOLO(
    "runs/detect/train-5/weights/best.pt"
)

camera = cv2.VideoCapture(0)


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera.")
        break

    result = model(
        frame,
        verbose=False
    )[0]

    display_frame = result.plot()

    cv2.imshow(
        "VisionOps PPE Test",
        display_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()