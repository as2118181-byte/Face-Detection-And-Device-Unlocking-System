import argparse
import csv
import os
import sys
import urllib
from glob import glob
from pathlib import Path
import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm
from facetools import FaceDetection
parser = argparse.ArgumentParser(description="Create named facebank CSV")
parser.add_argument("--images", type=str, required=True, help="Root folder containing sub-folders named after people (e.g. images/Arun/*.jpg)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to InceptionResnetV1_vggface2.onnx")
parser.add_argument("--output", type=str, required=True, help="Output CSV path (e.g. data/face_database.csv)")
args = parser.parse_args()
input_path = args.images
checkpoint_path = args.checkpoint
csv_path = args.output
if not os.path.isdir(input_path):
    print(f"The path [{input_path}] is not a directory")
    sys.exit(1)
person_dirs = [d for d in Path(input_path).iterdir() if d.is_dir()]
if not person_dirs:
    print("No person sub-folders found. Create folders like: images/Arun/, images/John/")
    sys.exit(1)
faceDetector = FaceDetection()

if not Path(checkpoint_path).is_file():
    print("Downloading InceptionResnetV1_vggface2.onnx ...")
    urllib.request.urlretrieve(
        "https://github.com/ffletcherr/face-recognition-liveness/releases/download/v0.1/InceptionResnetV1_vggface2.onnx",
        Path(checkpoint_path).absolute().as_posix(),
    )

resnet = onnxruntime.InferenceSession(checkpoint_path, providers=["CPUExecutionProvider"])

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    # Header: name + 512 embedding values
    writer.writerow(["name"] + [f"emb_{i}" for i in range(512)])

    for person_dir in person_dirs:
        name = person_dir.name
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))

        for image_path in tqdm(images, desc=f"Processing {name}"):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            faces, _ = faceDetector(image)
            if not faces:
                continue

            face_arr = faces[0]
            face_arr = np.moveaxis(face_arr, -1, 0)
            input_arr = np.expand_dims((face_arr - 127.5) / 128.0, 0)
            embeddings = resnet.run(["output"], {"input": input_arr.astype(np.float32)})[0]
            writer.writerow([name] + embeddings.flatten().tolist())

print(f"\nFacebank saved to: {csv_path}")