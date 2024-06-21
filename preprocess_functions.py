from datetime import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from skimage import transform as trans
from torchvision import transforms


def check_for_face_in_img(img: np.ndarray, keep_all: bool = True) -> Tuple[any, any, any]:
    """
    Check for faces in an image and return bounding boxes, probabilities, and landmarks.

    :param img: The input image in BGR format.
    :param keep_all: Whether to detect all faces (True) or just the one with the highest probability (False). Default
    is True.
    :return: A tuple containing:
        - boxes: The bounding boxes for detected faces.
        - probs: The probabilities associated with each detected face.
        - points: The facial landmarks for each detected face.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mtcnn = MTCNN(keep_all=keep_all, select_largest=False)
    boxes, probs, points = mtcnn.detect(img_rgb, landmarks=True)
    return boxes, probs, points


def align_face(img: np.ndarray, landmark: np.ndarray) -> np.ndarray:
    """
    Align a face in the image based on the provided facial landmarks.
    Function taken and adapted from
    https://github.com/av-savchenko/face-emotion-recognition/blob/main/src/VGAF_train.ipynb

    :param img: The input image in BGR format.
    :param landmark: The facial landmark coordinates for alignment.
    :return: The image with the aligned face.
    """
    image_size = [224, 224]

    # reference facial landmarks for a 224 x 224 image
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    src[:, 0] += 8.0
    src *= 2
    dst = landmark.astype(np.float32)

    # estimate the transformation matrix
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    matrix = tform.params[0:2, :]

    # apply the transformation to align the face
    warped = cv2.warpAffine(img, matrix, (image_size[1], image_size[0]), borderValue=0.0)

    return warped


def align_and_transform_img(img: np.ndarray,
                            landmark: np.ndarray,
                            transform: transforms.Compose = None,
                            img_size: int = 224) -> Tuple[Image.Image, torch.Tensor]:
    """
    Align and transform an image based on facial landmarks.

    :param img: The input image in BGR format.
    :param landmark: The facial landmark coordinates
    :param transform: The transformation pipeline to apply to the image. If None, a default transformation pipeline
    is used.
    :param img_size: The desired image size. Default is 224.
    :return: A tuple containing:
        - img_pil: The aligned image as a PIL image.
        - transformed: the transformed image as a tensor.
    """
    aligned = align_face(img, landmark)
    img_pil = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )

    transformed = transform(img_pil).unsqueeze(dim=0)

    return img_pil, transformed


def extract_frames(video_path: Path, frame_interval: int = 1) -> Dict[str, any]:
    """
    Extract frames from a video and save them as images in the specified output folder.

    :param video_path: The path to the input video file.
    :param frame_interval: The interval between frames to be extracted. Default is 1.
    :return: A dictionary containing the number of frames extracted and the output folder path.
    """
    # create the output folder if it doesn't exist
    now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = video_path.parent / f"{now}_{video_path.stem}_frames"
    output_folder.mkdir(exist_ok=True)

    # open the video file
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    extracted_frame_count = 0

    # read and save frames until the end of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # save the frame as an image
        if frame_count % frame_interval == 0:
            frame_path = output_folder / f"frame_{frame_count}.png"
            cv2.imwrite(str(frame_path), frame)
            extracted_frame_count += 1

    # release the video capture object
    cap.release()

    return {"frame_count": extracted_frame_count, "folder_name": str(output_folder)}
