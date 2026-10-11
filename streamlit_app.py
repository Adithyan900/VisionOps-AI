import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="VisionOps AI",
    page_icon="👷",
    layout="wide"
)


st.title("VisionOps AI")
st.caption(
    "AI-powered visual inspection prototype using "
    "object detection and custom PPE analysis."
)


# =========================================================
# MODEL SETTINGS
# =========================================================

GENERAL_MODEL_PATH = "yolo11n.pt"
PPE_MODEL_PATH = "models/ppe_best.pt"

PPE_CONF_THRESHOLD = 0.25


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    general_model = YOLO(
        GENERAL_MODEL_PATH
    )

    ppe_model = YOLO(
        PPE_MODEL_PATH
    )

    return general_model, ppe_model


try:

    general_model, ppe_model = load_models()

except Exception as error:

    st.error(
        f"Model loading failed: {error}"
    )

    st.stop()


# =========================================================
# PPE SUMMARY
# =========================================================

def build_ppe_summary(detections):

    scores = {}

    for name, confidence in detections:

        name = name.lower()

        scores[name] = max(
            scores.get(name, 0.0),
            confidence
        )


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


        if (
            negative >= PPE_CONF_THRESHOLD
            and negative > positive
        ):

            return "MISSING", negative


        if positive >= PPE_CONF_THRESHOLD:

            return "DETECTED", positive


        return (
            "NOT CONFIRMED",
            max(
                positive,
                negative
            )
        )


    summary = []


    helmet_status, helmet_conf = check_pair(
        "helmet",
        "no_helmet"
    )

    summary.append(
        (
            "Helmet",
            helmet_status,
            helmet_conf
        )
    )


    gloves_status, gloves_conf = check_pair(
        "gloves",
        "no_gloves"
    )

    summary.append(
        (
            "Gloves",
            gloves_status,
            gloves_conf
        )
    )


    boots_status, boots_conf = check_pair(
        "boots",
        "no_boots"
    )

    summary.append(
        (
            "Boots",
            boots_status,
            boots_conf
        )
    )


    goggles_status, goggles_conf = check_pair(
        "goggles",
        "no_goggle"
    )

    summary.append(
        (
            "Goggles",
            goggles_status,
            goggles_conf
        )
    )


    vest_conf = scores.get(
        "vest",
        0.0
    )


    if vest_conf >= PPE_CONF_THRESHOLD:

        vest_status = "DETECTED"

    else:

        vest_status = "NOT CONFIRMED"


    summary.append(
        (
            "Vest",
            vest_status,
            vest_conf
        )
    )


    statuses = [
        item[1]
        for item in summary
    ]


    if "MISSING" in statuses:

        overall = "REVIEW REQUIRED"


    elif "NOT CONFIRMED" in statuses:

        overall = (
            "MANUAL VERIFICATION REQUIRED"
        )


    else:

        overall = (
            "PPE DETECTED - VERIFY"
        )


    return summary, overall


# =========================================================
# IMAGE INPUT
# =========================================================

st.subheader(
    "Capture or upload an inspection image"
)


tab1, tab2 = st.tabs(
    [
        "📷 Camera",
        "📁 Upload Image"
    ]
)


with tab1:

    camera_image = st.camera_input(
        "Take a photo"
    )


with tab2:

    uploaded_image = st.file_uploader(
        "Upload JPG, JPEG or PNG",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


input_file = None


if camera_image is not None:

    input_file = camera_image


elif uploaded_image is not None:

    input_file = uploaded_image


# =========================================================
# PROCESS IMAGE
# =========================================================

if input_file is not None:

    file_bytes = np.asarray(
        bytearray(
            input_file.read()
        ),
        dtype=np.uint8
    )


    frame = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


    if frame is None:

        st.error(
            "Could not read image."
        )

        st.stop()


    st.divider()

    st.subheader(
        "VisionOps Inspection"
    )


    # =====================================================
    # GENERAL YOLO
    # =====================================================

    general_result = general_model(
        frame,
        verbose=False
    )[0]


    general_annotated = (
        general_result.plot()
    )


    # =====================================================
    # FIND PERSON DETECTIONS
    # =====================================================

    persons = []


    for box in general_result.boxes:

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


        if object_name == "person":

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            persons.append(
                (
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


    # =====================================================
    # UI COLUMNS
    # =====================================================

    col1, col2 = st.columns(
        [1.4, 1]
    )


    with col1:

        st.markdown(
            "### General Object Detection"
        )


        general_rgb = cv2.cvtColor(
            general_annotated,
            cv2.COLOR_BGR2RGB
        )


        st.image(
            general_rgb,
            use_container_width=True
        )


    # =====================================================
    # PPE INSPECTION
    # =====================================================

    with col2:

        st.markdown(
            "### PPE Inspection"
        )


        if not persons:

            st.warning(
                "No person was detected "
                "in this image."
            )


        else:

            # Select highest-confidence person
            persons.sort(
                key=lambda item: item[0],
                reverse=True
            )


            (
                person_conf,
                x1,
                y1,
                x2,
                y2
            ) = persons[0]


            height, width, _ = (
                frame.shape
            )


            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                width,
                x2
            )

            y2 = min(
                height,
                y2
            )


            person_crop = frame[
                y1:y2,
                x1:x2
            ]


            if person_crop.size == 0:

                st.error(
                    "Could not create "
                    "person inspection region."
                )


            else:

                ppe_result = ppe_model(
                    person_crop,
                    verbose=False,
                    conf=PPE_CONF_THRESHOLD
                )[0]


                ppe_annotated = (
                    ppe_result.plot()
                )


                ppe_rgb = cv2.cvtColor(
                    ppe_annotated,
                    cv2.COLOR_BGR2RGB
                )


                st.image(
                    ppe_rgb,
                    caption="Selected Person ROI",
                    use_container_width=True
                )


                detections = []


                for box in ppe_result.boxes:

                    class_id = int(
                        box.cls[0]
                    )


                    confidence = float(
                        box.conf[0]
                    )


                    name = (
                        ppe_model.names[
                            class_id
                        ]
                    )


                    if (
                        name.lower()
                        not in
                        [
                            "person",
                            "none"
                        ]
                    ):

                        detections.append(
                            (
                                name,
                                confidence
                            )
                        )


                summary, overall = (
                    build_ppe_summary(
                        detections
                    )
                )


                st.metric(
                    "Person Detection Confidence",
                    f"{person_conf:.2f}"
                )


                st.markdown(
                    "#### PPE Snapshot Analysis"
                )


                for (
                    item,
                    status,
                    confidence
                ) in summary:


                    if status == "DETECTED":

                        icon = "✅"


                    elif status == "MISSING":

                        icon = "❌"


                    else:

                        icon = "⚠️"


                    if confidence > 0:

                        st.write(
                            f"{icon} **{item}:** "
                            f"{status} "
                            f"({confidence:.2f})"
                        )


                    else:

                        st.write(
                            f"{icon} **{item}:** "
                            f"{status}"
                        )


                st.divider()


                if overall == "REVIEW REQUIRED":

                    st.error(
                        f"Status: {overall}"
                    )


                elif (
                    overall
                    ==
                    "MANUAL VERIFICATION REQUIRED"
                ):

                    st.warning(
                        f"Status: {overall}"
                    )


                else:

                    st.success(
                        f"Status: {overall}"
                    )


                st.caption(
                    "Prototype model — "
                    "human verification required."
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "VisionOps AI V1 | "
    "General YOLO + Custom PPE Detection"
)