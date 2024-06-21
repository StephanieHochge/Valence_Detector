from datetime import datetime as dt
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import torch

import engine as e
import preprocess_functions as pf


def create_save_dir(predict_dir: Path, filename: str) -> Path:
    """
    Create a directory for saving prediction results. The directory name includes a timestamp and the filename.

    :param predict_dir: The base directory where the new directory will be created.
    :param filename: The name of the file.
    :return: The path to the newly created directory.
    """
    now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_dir = predict_dir / f"{now}_{filename}"
    new_dir.mkdir(exist_ok=True)
    return new_dir


def process_videos(uploaded: str,
                   left_col: st.columns,
                   right_col: st.columns,
                   val_model: torch.nn.Module,
                   predict_dir: Path,
                   filename: str) -> pd.DataFrame:
    """
    Process an uploaded video file to detect faces and analyze valence.

    :param uploaded: The path to the video file.
    :param left_col: The Streamlit left column for displaying the video.
    :param right_col: The Streamlit right column for displaying valence results.
    :param val_model: The model to use for valence prediction.
    :param predict_dir: The path to the valence predictions.
    :param filename: The name of the uploaded file.
    :return: A DataFrame containing the analysis results.
    """
    left_col.video(uploaded)
    right_col.write(f"Extracting frames... Please wait!")
    meta_data = pf.extract_frames(Path(uploaded), frame_interval=1)
    right_col.write(f"Extracted {meta_data['frame_count']} frames. Frames were saved to 'uploaded_files' with "
                    f"the current timestamp.")

    # check if there is at least one frame with a face on it
    folder_name = Path(meta_data["folder_name"])
    img_list = list(folder_name.glob("*.png"))
    new_dir = create_save_dir(predict_dir, filename)

    val_dict = {}
    face_count = 0
    face_detected = False

    for img_path in img_list:
        img = cv2.imread(str(img_path))
        _, probs, landmarks = pf.check_for_face_in_img(img, keep_all=True)
        frame = img_path.stem

        if landmarks is not None:
            # keep only face detections with a probability of at least 0.95
            hp_landmarks = [landmark for prob, landmark in zip(probs, landmarks) if prob > 0.95]

            if hp_landmarks:
                if not face_detected:
                    right_col.write("There is at least one frame depicting a face.")
                    face_detected = True
                    right_col.write("Detecting valence... Please wait!")
                face_count += len(hp_landmarks)
                save_dir = new_dir / img_path.stem if len(hp_landmarks) > 1 else new_dir
                save_dir.mkdir(exist_ok=True)

                for j, landmark in enumerate(hp_landmarks):
                    img_pil, transformed_image = pf.align_and_transform_img(img, landmark)
                    valence = e.predict_valence(transformed_image, val_model)
                    img_pil.save(save_dir / f"{img_path.stem}_{j + 1}_val_{valence:.3f}.jpg")
                    val_dict[f"{frame}_{j + 1}"] = valence

    if face_count > 0:
        val_df = pd.DataFrame(list(val_dict.items()), columns=["frame number", "valence"])
        csv_path = new_dir / "valence_table.csv"
        val_df.to_csv(str(csv_path), index=False, sep=";")
        right_column.write(
            f"Analysis results were downloaded into "
            f"'uploaded_files/predicted_valence/videos' with"
            f" the current timestamp.")
        return val_df
    else:
        right_col.write("No faces were detected. Please upload another file.")
        return pd.DataFrame({
            "Feature": ["file type", "number of faces"],
            "Properties": ["video", 0]
        })


def process_image(uploaded: str,
                  left_col: st.columns,
                  right_col: st.columns,
                  val_model: torch.nn.Module,
                  predict_dir: Path,
                  filename: str) -> pd.DataFrame:
    """
    Process an uploaded image file to detect faces and analyze valence.

    :param uploaded: The path to the image file.
    :param left_col: The Streamlit left column for displaying the image.
    :param right_col: The Streamlit right column for displaying valence results.
    :param val_model: The model to use for valence prediction.
    :param predict_dir: The path to the valence predictions.
    :param filename: The name of the uploaded file.
    :return: A DataFrame containing the analysis results.
    """
    left_col.image(uploaded)
    img = cv2.imread(uploaded)
    _, probs, landmarks = pf.check_for_face_in_img(img, keep_all=True)
    no_face_message = "No faces were detected. Please upload another file."
    no_face_df = pd.DataFrame({
        "Feature": ["file type", "number of faces"],
        "Properties": ["image", 0]
    })

    if landmarks is None:
        right_col.write(no_face_message)
        return no_face_df
    else:
        hp_landmarks = [landmark for prob, landmark in zip(probs, landmarks) if prob > 0.95]

        if not hp_landmarks:
            right_col.write(no_face_message)
            return no_face_df
        else:
            no_faces = len(hp_landmarks)
            right_col.write(f"{no_faces} face/faces was/were detected. Analyzing valence....")

            save_dir = create_save_dir(predict_dir, filename)
            val_dict = {}
            for i, landmark in enumerate(landmarks):
                img_pil, transformed_image = pf.align_and_transform_img(img, landmark)
                valence = e.predict_valence(transformed_image, val_model)
                img_pil.save(f"{save_dir}/{i + 1}_val_{valence:.3f}.jpg")
                val_dict[i + 1] = valence

            val_df = pd.DataFrame(list(val_dict.items()), columns=["face number", "valence"])
            csv_path = save_dir / "valence_table.csv"
            val_df.to_csv(str(csv_path), index=False, sep=";")
            right_column.write(
                f"Analysis results were downloaded into "
                f"'uploaded_files/predicted_valence/images' with"
                f" the current timestamp.")
            return val_df


if __name__ == "__main__":
    # setup folder structure
    upload_dir = Path("uploaded_files")
    predicted_dir = upload_dir / "predicted_valence"
    video_dir = predicted_dir / "videos"
    image_dir = predicted_dir / "images"
    for folder in [upload_dir, predicted_dir, video_dir, image_dir]:
        folder.mkdir(exist_ok=True)

    # load the model for evaluation
    if 'model' not in st.session_state:
        st.session_state.model = e.load_model_for_eval(
            architecture_path="pretrained_models/enet_b0_8_best_vgaf.pt",
            state_dict_path="pretrained_models/final_enet.pth",
            dropout=0.28882)

    # set the title and introductory text of the app
    st.title("Valence Detector")
    st.write("Upload a video or image of a face and find out what valence the facial expression shows!")

    # divide the screen into a left and a right column
    left_column, right_column = st.columns(2)

    # define the elements of the left column
    left_column.header("File Preview")
    uploaded_file = left_column.file_uploader("Choose image or video:",
                                              type=["png", "jpg", "mp4"])

    # define the elements of the right column
    right_column.header("Analysis Results")

    df_result = pd.DataFrame({
        "face number": [1],
        "valence": [0.0633]
    })

    # process the uploaded file
    if uploaded_file:
        # save the uploaded file
        file_name = uploaded_file.name
        uploaded_path = upload_dir / file_name

        with open(uploaded_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # check the file type and process the file accordingly
        file_type = uploaded_file.name.split(".")[-1].lower()  # check file type
        if file_type in ["mp4"]:
            df_result = process_videos(uploaded=str(uploaded_path),
                                       left_col=left_column,
                                       right_col=right_column,
                                       val_model=st.session_state.model,
                                       predict_dir=video_dir,
                                       filename=file_name)
        elif file_type in ["jpg", "png", "jpeg"]:
            df_result = process_image(uploaded=str(uploaded_path),
                                      left_col=left_column,
                                      right_col=right_column,
                                      val_model=st.session_state.model,
                                      predict_dir=image_dir,
                                      filename=file_name)

    else:  # if no file was uploaded, show the default picture
        left_column.image("media/male_face.jpg")

    right_column.dataframe(df_result, hide_index=True)
