import cv2
import json

from config import CASC_PATH


def load_tensors(path):
    """
    load tensors from json file
    :param path: tensor file path
    :return: tensors
    """
    with open(path, encoding='utf-8') as f:
        return json.loads(f.read())


cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is no face found in image
    if not len(faces) > 0:
        return None, None
    max_are_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face
    # face to image
    face_edge = max_are_face
    image = image[face_edge[1]:(face_edge[1] + face_edge[2]), face_edge[0]:(face_edge[0] + face_edge[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("[+} Problem during resize")
        return None, None
    return image, face_edge


def format_image_rgb(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is no face found in image
    if not len(faces) > 0:
        return None, None
    max_are_face = faces[0]
    print('Max Face', faces[0])
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face
    # face to image
    face_edge = max_are_face
    image = image[face_edge[1]:(face_edge[1] + face_edge[2]), face_edge[0]:(face_edge[0] + face_edge[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (240, 240), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("[+} Problem during resize")
        return None, None
    edge = (face_edge[0], face_edge[1], face_edge[0] + face_edge[3], face_edge[1] + face_edge[2])
    return image, edge
