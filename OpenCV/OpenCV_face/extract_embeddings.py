from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
from tomlkit import parse


def load_config(toml_file, table):
    with open(toml_file, "r") as f:
        data = f.read()
        config = parse(data)
        print(config)
    return config[table]


if __name__ == "__main__":
    config = load_config("config.toml", "embeddingsExtractor")
    detector = cv2.dnn.readNetFromCaffe(config["detector_path"], config["weights_path"])
    embedder = cv2.dnn.readNetFromTorch(config["embedding_model"])
    ImagePaths = list(paths.list_images(config["training_directory"]))
    output_dir = config["face_embeddings_directory"]
    known_embeddings = []
    known_names = []
    total = 0
    for (i, imagePath) in enumerate(ImagePaths):
        print("[INFO] processing image {}/{}".format(i + 1, len(ImagePaths)))
        label = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )

        # apply OpenCV's deep learning-based detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()
        # ensure at least one face was found
        if len(detections) > 0:
            # assuming only one face
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # meets our minimum probability test (thus helping filter out weak
            # dections)

            if confidence > float(config["minimum_confidence"]):
                # create bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract face ROI
                face = image[startY:endY, startX:endX]
                # grab ROI dimensions
                (face_width, face_height) = face.shape[:2]

                if face_width < 20 or face_height < 20:
                    continue

                # construct a blob for the face ROI, then pass
                # through embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(
                    face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
                )
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face
                known_embeddings.append(vec.flatten())
                total += 1

    # dump the facial embeddings
    print("".format(total))
    data = {"embeddings": known_embeddings, "names": known_names}
    with open(output_dir + "embeddings.pickle", "wb") as f:
        f.write(pickle.dumps(data))
