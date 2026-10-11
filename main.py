import cv2
import mediapipe as mp
import math
from pathlib import Path
from glob import glob
from ultralytics import YOLO


# =========================================================
# VISIONOPS AI - V1
# Touchless Visual Inspection Prototype
# =========================================================


# =========================================================
# 1. SETTINGS
# =========================================================

GENERAL_MODEL_PATH = "yolo11n.pt"

# Minimum confidence accepted from PPE model
PPE_CONF_THRESHOLD = 0.25


# =========================================================
# 2. FIND PPE MODEL
# =========================================================

def find_ppe_model():

    possible_paths = [
        "models/ppe_best.pt",
        "runs/detect/train-5/weights/best.pt"
    ]

    # First check known locations
    for path in possible_paths:

        if Path(path).exists():
            return path


    # If not found, search all YOLO training folders
    run_models = glob(
        "runs/detect/train*/weights/best.pt"
    )


    if run_models:

        # Use most recently created model
        run_models.sort(
            key=lambda p: Path(p).stat().st_mtime,
            reverse=True
        )

        return run_models[0]


    raise FileNotFoundError(
        "\nPPE model not found.\n"
        "Keep your best.pt inside:\n"
        "runs/detect/train-5/weights/best.pt\n"
    )


# =========================================================
# 3. LOAD GENERAL YOLO MODEL
# =========================================================

general_model = YOLO(
    GENERAL_MODEL_PATH
)


# =========================================================
# 4. LOAD CUSTOM PPE MODEL
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
# 5. MEDIAPIPE HAND SETUP
# =========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


# =========================================================
# 6. INSPECTION ROUTER
# =========================================================

def choose_inspection(object_name):

    if object_name == "person":

        return "PPE CHECK"


    elif object_name in [
        "cell phone",
        "laptop"
    ]:

        return "DEVICE VISUAL REVIEW"


    elif object_name in [
        "bottle",
        "cup"
    ]:

        return "CONTAINER VISUAL REVIEW"


    else:

        return "GENERAL VISUAL REVIEW"


# =========================================================
# 7. PPE RESULT ANALYSIS
# =========================================================

def build_ppe_summary(detections):

    scores = {}


    # Keep highest confidence
    # for every PPE class
    for name, confidence in detections:

        name = name.lower()

        scores[name] = max(
            scores.get(name, 0.0),
            confidence
        )


    # -----------------------------------------------------
    # Compare PPE class vs missing-PPE class
    # -----------------------------------------------------

    def check_pair(
        positive_name,
        negative_name
    ):

        positive = scores.get(
            positive_name,
            0.0
        )

        negative = scores.get(
            negative_name,
            0.0
        )


        # Example:
        # no_helmet stronger than helmet
        if (
            negative >= PPE_CONF_THRESHOLD
            and negative > positive
        ):

            return "MISSING", negative


        if positive >= PPE_CONF_THRESHOLD:

            return "DETECTED", positive


        # No strong result either way
        return (
            "NOT CONFIRMED",
            max(
                positive,
                negative
            )
        )


    summary = []


    # -----------------------------------------------------
    # HELMET
    # -----------------------------------------------------

    status, confidence = check_pair(
        "helmet",
        "no_helmet"
    )

    summary.append(
        (
            "HELMET",
            status,
            confidence
        )
    )


    # -----------------------------------------------------
    # GLOVES
    # -----------------------------------------------------

    status, confidence = check_pair(
        "gloves",
        "no_gloves"
    )

    summary.append(
        (
            "GLOVES",
            status,
            confidence
        )
    )


    # -----------------------------------------------------
    # BOOTS
    # -----------------------------------------------------

    status, confidence = check_pair(
        "boots",
        "no_boots"
    )

    summary.append(
        (
            "BOOTS",
            status,
            confidence
        )
    )


    # -----------------------------------------------------
    # GOGGLES
    # -----------------------------------------------------

    status, confidence = check_pair(
        "goggles",
        "no_goggle"
    )

    summary.append(
        (
            "GOGGLES",
            status,
            confidence
        )
    )


    # -----------------------------------------------------
    # VEST
    #
    # Dataset doesn't have a no_vest class,
    # so absence is NOT automatically declared.
    # -----------------------------------------------------

    vest_confidence = scores.get(
        "vest",
        0.0
    )


    if vest_confidence >= PPE_CONF_THRESHOLD:

        vest_status = "DETECTED"

    else:

        vest_status = "NOT CONFIRMED"


    summary.append(
        (
            "VEST",
            vest_status,
            vest_confidence
        )
    )


    # -----------------------------------------------------
    # OVERALL STATUS
    # -----------------------------------------------------

    statuses = [
        item[1]
        for item in summary
    ]


    if "MISSING" in statuses:

        overall_status = (
            "REVIEW REQUIRED"
        )


    elif "NOT CONFIRMED" in statuses:

        overall_status = (
            "MANUAL VERIFICATION REQUIRED"
        )


    else:

        overall_status = (
            "PPE DETECTED - VERIFY"
        )


    return (
        summary,
        overall_status
    )


# =========================================================
# 8. CAMERA SETUP
# =========================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    raise RuntimeError(
        "Could not open camera."
    )


# =========================================================
# 9. STATE VARIABLES
# =========================================================

selected_point = None

selected_object = None

selected_confidence = 0.0

selected_bbox = None


pinch_active = False


ppe_detections = []

ppe_summary = []

ppe_status = ""

ppe_checked = False


# =========================================================
# 10. MAIN LOOP
# =========================================================

while True:


    # -----------------------------------------------------
    # READ CAMERA
    # -----------------------------------------------------

    success, frame = camera.read()


    if not success:

        print(
            "Could not read camera."
        )

        break


    # True only on the frame
    # where a NEW pinch happens
    selection_event = False


    # =====================================================
    # 11. GENERAL OBJECT DETECTION
    # =====================================================

    yolo_result = general_model(
        frame,
        verbose=False
    )[0]


    # YOLO annotated frame
    display_frame = yolo_result.plot()


    # =====================================================
    # 12. MEDIAPIPE HAND DETECTION
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    hand_result = hands.process(
        rgb_frame
    )


    # =====================================================
    # 13. HAND TRACKING
    # =====================================================

    if hand_result.multi_hand_landmarks:


        for hand_landmarks in (
            hand_result.multi_hand_landmarks
        ):


            height, width, _ = (
                frame.shape
            )


            # ---------------------------------------------
            # INDEX TIP
            # Landmark 8
            # ---------------------------------------------

            index_tip = (
                hand_landmarks.landmark[8]
            )


            # ---------------------------------------------
            # THUMB TIP
            # Landmark 4
            # ---------------------------------------------

            thumb_tip = (
                hand_landmarks.landmark[4]
            )


            # Convert index coordinates
            # from normalized values
            # to pixel coordinates
            x = int(
                index_tip.x * width
            )

            y = int(
                index_tip.y * height
            )


            # ---------------------------------------------
            # PINCH DISTANCE
            # ---------------------------------------------

            pinch_distance = math.hypot(

                index_tip.x
                -
                thumb_tip.x,

                index_tip.y
                -
                thumb_tip.y
            )


            # =================================================
            # 14. PINCH = SELECT
            # =================================================

            if (
                pinch_distance < 0.05
                and not pinch_active
            ):

                selected_point = (
                    x,
                    y
                )

                pinch_active = True

                selection_event = True


            elif pinch_distance > 0.07:

                pinch_active = False


            # =================================================
            # 15. POINTING GESTURE
            # =================================================

            index_up = (

                hand_landmarks.landmark[8].y
                <
                hand_landmarks.landmark[6].y
            )


            middle_down = (

                hand_landmarks.landmark[12].y
                >
                hand_landmarks.landmark[10].y
            )


            ring_down = (

                hand_landmarks.landmark[16].y
                >
                hand_landmarks.landmark[14].y
            )


            pinky_down = (

                hand_landmarks.landmark[20].y
                >
                hand_landmarks.landmark[18].y
            )


            if (
                index_up
                and middle_down
                and ring_down
                and pinky_down
            ):

                gesture = "POINTING"

            else:

                gesture = "HAND"


            # =================================================
            # 16. TARGET CROSSHAIR
            # =================================================

            if gesture == "POINTING":


                # Circle
                cv2.circle(
                    display_frame,
                    (x, y),
                    35,
                    (0, 225, 225),
                    2
                )


                # Horizontal line
                cv2.line(
                    display_frame,
                    (x - 50, y),
                    (x + 50, y),
                    (0, 225, 225),
                    2
                )


                # Vertical line
                cv2.line(
                    display_frame,
                    (x, y - 50),
                    (x, y + 50),
                    (0, 225, 225),
                    2
                )


                cv2.putText(
                    display_frame,
                    "TARGET",
                    (
                        x + 40,
                        y - 40
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 225, 225),
                    2
                )


            # =================================================
            # 17. INDEX FINGER MARKER
            # =================================================

            cv2.circle(
                display_frame,
                (x, y),
                10,
                (0, 0, 225),
                -1
            )


            cv2.putText(
                display_frame,
                f"Index: {x}, {y}",
                (
                    x + 18,
                    y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 225, 0),
                2
            )


            # Draw skeleton
            mp_draw.draw_landmarks(
                display_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


    # =====================================================
    # 18. PROCESS NEW SELECTION
    # =====================================================

    if (
        selection_event
        and selected_point is not None
    ):


        sx, sy = selected_point


        matching_objects = []


        # Search YOLO objects
        for box in yolo_result.boxes:


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Is selected point
            # inside this box?
            if (
                x1 <= sx <= x2
                and
                y1 <= sy <= y2
            ):


                class_id = int(
                    box.cls[0]
                )


                confidence = float(
                    box.conf[0]
                )


                object_name = (
                    general_model.names[
                        class_id
                    ]
                )


                # Bounding box area
                area = (

                    (x2 - x1)
                    *
                    (y2 - y1)
                )


                matching_objects.append(
                    (
                        area,
                        object_name,
                        confidence,
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )
                )


        # ---------------------------------------------
        # Clear old results
        # ---------------------------------------------

        selected_object = None

        selected_confidence = 0.0

        selected_bbox = None


        ppe_detections = []

        ppe_summary = []

        ppe_status = ""

        ppe_checked = False


        # ---------------------------------------------
        # Select most specific object
        # ---------------------------------------------

        if matching_objects:


            matching_objects.sort(
                key=lambda item: item[0]
            )


            (
                _,
                selected_object,
                selected_confidence,
                selected_bbox
            ) = matching_objects[0]


        # =================================================
        # 19. PPE INSPECTION
        # =================================================

        if (
            selected_object == "person"
            and
            selected_bbox is not None
        ):


            (
                x1,
                y1,
                x2,
                y2
            ) = selected_bbox


            (
                frame_height,
                frame_width,
                _
            ) = frame.shape


            # Prevent invalid coordinates
            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                frame_width,
                x2
            )

            y2 = min(
                frame_height,
                y2
            )


            # ---------------------------------------------
            # PERSON ROI
            # ---------------------------------------------

            person_crop = frame[
                y1:y2,
                x1:x2
            ]


            if person_crop.size > 0:


                # -----------------------------------------
                # RUN CUSTOM PPE MODEL
                # -----------------------------------------

                ppe_result = ppe_model(

                    person_crop,

                    verbose=False,

                    conf=PPE_CONF_THRESHOLD

                )[0]


                ppe_checked = True


                # -----------------------------------------
                # READ PPE DETECTIONS
                # -----------------------------------------

                for box in ppe_result.boxes:


                    class_id = int(
                        box.cls[0]
                    )


                    confidence = float(
                        box.conf[0]
                    )


                    ppe_name = (
                        ppe_model.names[
                            class_id
                        ]
                    )


                    # Ignore Person
                    # because general YOLO
                    # already selected the person
                    if (
                        ppe_name.lower()
                        not in
                        [
                            "person",
                            "none"
                        ]
                    ):

                        ppe_detections.append(
                            (
                                ppe_name,
                                confidence
                            )
                        )


                # Build clean PPE report
                (
                    ppe_summary,
                    ppe_status
                ) = build_ppe_summary(
                    ppe_detections
                )


    # =====================================================
    # 20. DRAW SELECTED POINT
    # =====================================================

    if selected_point is not None:


        cv2.circle(
            display_frame,
            selected_point,
            25,
            (255, 0, 255),
            3
        )


        cv2.putText(
            display_frame,
            "SELECTED",
            (
                selected_point[0] + 30,
                selected_point[1]
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )


    # =====================================================
    # 21. HIGHLIGHT SELECTED OBJECT
    # =====================================================

    if selected_bbox is not None:


        (
            x1,
            y1,
            x2,
            y2
        ) = selected_bbox


        cv2.rectangle(
            display_frame,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            (255, 0, 255),
            3
        )


    # =====================================================
    # 22. DISPLAY OBJECT INFORMATION
    # =====================================================

    if selected_object is not None:


        inspection_type = choose_inspection(
            selected_object
        )


        cv2.putText(
            display_frame,
            f"OBJECT: {selected_object}",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )


        cv2.putText(
            display_frame,
            f"CONFIDENCE: {selected_confidence:.2f}",
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        cv2.putText(
            display_frame,
            f"INSPECTION: {inspection_type}",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )


        # =================================================
        # 23. PPE RESULT DISPLAY
        # =================================================

        if selected_object == "person":


            cv2.putText(
                display_frame,
                "PPE SNAPSHOT ANALYSIS",
                (30, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            if ppe_checked:


                text_y = 215


                for (
                    label,
                    status,
                    confidence
                ) in ppe_summary:


                    if confidence > 0:

                        result_text = (
                            f"{label}: "
                            f"{status} "
                            f"({confidence:.2f})"
                        )

                    else:

                        result_text = (
                            f"{label}: "
                            f"{status}"
                        )


                    cv2.putText(
                        display_frame,
                        result_text,
                        (
                            30,
                            text_y
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2
                    )


                    text_y += 25


                # -----------------------------------------
                # OVERALL PPE STATUS
                # -----------------------------------------

                cv2.putText(
                    display_frame,
                    f"STATUS: {ppe_status}",
                    (
                        30,
                        text_y + 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2
                )


                # -----------------------------------------
                # SAFETY MESSAGE
                # -----------------------------------------

                cv2.putText(
                    display_frame,
                    "Prototype model - human verification required",
                    (
                        30,
                        text_y + 40
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 165, 255),
                    1
                )


    # =====================================================
    # 24. NO OBJECT SELECTED
    # =====================================================

    elif selected_point is not None:


        cv2.putText(
            display_frame,
            "No detected object at selected point",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )


    # =====================================================
    # 25. LOW LIGHT DETECTION
    # =====================================================

    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    brightness = (
        gray_frame.mean()
    )


    if brightness < 45:


        cv2.putText(
            display_frame,
            "LOW LIGHT - RESULTS MAY BE UNRELIABLE",
            (
                30,
                display_frame.shape[0] - 45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2
        )


    # =====================================================
    # 26. CONTROLS
    # =====================================================

    cv2.putText(
        display_frame,
        "Q: Quit   R: Reset selection",
        (
            30,
            display_frame.shape[0] - 15
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    # =====================================================
    # 27. DISPLAY
    # =====================================================

    cv2.imshow(
        "VisionOps AI - V1",
        display_frame
    )


    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    # Quit
    if key == ord("q"):

        break


    # Reset
    if key == ord("r"):


        selected_point = None

        selected_object = None

        selected_confidence = 0.0

        selected_bbox = None


        ppe_detections = []

        ppe_summary = []

        ppe_status = ""

        ppe_checked = False


# =========================================================
# 28. CLEANUP
# =========================================================

camera.release()

hands.close()

cv2.destroyAllWindows()