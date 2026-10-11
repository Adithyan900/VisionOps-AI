from ultralytics import YOLO


# =========================================================
# LOAD BASE YOLO MODEL
# =========================================================

model = YOLO(
    "yolo11n.pt"
)


# =========================================================
# TRAIN PPE MODEL
# =========================================================

results = model.train(

    data="construction-ppe.yaml",

    epochs=1,

    imgsz=416,

    batch=2,

    device="mps"

)


print(
    "PPE prototype training completed."
)