import tensorflow as tf
import tensorflow_hub as hub
import cv2


img_path = r'images/ball.jpg'

img = cv2.imread(img_path)
dim = (320,320)

image_tensor = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# Apply image detector on a single image.
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
detector_output = detector(tf.expand_dims(image_tensor,axis=0))
class_ids = detector_output["detection_classes"]

print('classes_ids')