import cv2

from pathlib import Path

from glob import glob

from ultralytics import YOLO


# =========================================================
# FIND PPE MODEL
# =========================================================

def find_ppe_model():

    possible_paths = [

        "models/ppe_best.pt",

        "runs/detect/train-5/weights/best.pt"

    ]


    for path in possible_paths:

        if Path(path).exists():

            return path


    run_models = glob(
        "runs/detect/train*/weights/best.pt"
    )


    if run_models:

        run_models.sort(

            key=lambda p:
            Path(p).stat().st_mtime,

            reverse=True
        )


        return run_models[0]


    raise FileNotFoundError(
        "No PPE best.pt model found."
    )


# =========================================================
# LOAD PPE MODEL
# =========================================================

ppe_model_path = find_ppe_model()


print(
    "Using PPE model:",
    ppe_model_path
)


ppe_model = YOLO(
    ppe_model_path
)


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)


# =========================================================
# MAIN LOOP
# =========================================================

while True:


    success, frame = camera.read()


    if not success:

        print(
            "Could not read camera."
        )

        break


    # PPE detection
    result = ppe_model(
        frame,
        verbose=False
    )[0]


    # Draw detections
    display_frame = result.plot()


    cv2.imshow(
        "VisionOps PPE Test",
        display_frame
    )


    # q = quit
    if (
        cv2.waitKey(1)
        &
        0xFF
        ==
        ord("q")
    ):

        break


# =========================================================
# CLEANUP
# =========================================================

camera.release()

cv2.destroyAllWindows()