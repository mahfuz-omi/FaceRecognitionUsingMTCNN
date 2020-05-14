from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import PIL.Image as Image

# https://www.sitepoint.com/keras-face-detection-recognition/

# image_pil = Image.open('horse.jpg')
# print('PIL image',image_pil)
#
# plt.imshow(image_pil)
# plt.show()
#
# # get the numpy array of the image
# image_matplot = plt.imread('horse.jpg')
# print('matplotlib image',image_matplot)
# print(type(image_matplot))
# # <class 'numpy.ndarray'>
# print(image_matplot.shape)
# # (28, 28, 3)
#
# plt.imshow(image_matplot)
# plt.show()

image1 = plt.imread('face_recognition.jpg')
image2 = plt.imread('face_recognition3.jpg')
print(image1.shape)
detector = MTCNN()

faces1 = detector.detect_faces(image1)
for face in faces1:
    print(face)

faces2 = detector.detect_faces(image2)
for face in faces2:
    print(face)

plt.imshow(image1)
plt.show()

plt.imshow(image2)
plt.show()

# {'box': [151, 332, 66, 80],
# 'confidence': 0.999945878982544,
# 'keypoints': {'left_eye': (163, 362), 'right_eye': (192, 362),
# 'nose': (172, 375), 'mouth_left': (163, 392), 'mouth_right': (191, 392)}}

#show face highlighted
from matplotlib.patches import Rectangle
def highlight_faces(image, faces):
    #display image
    plt.imshow(image)

    # gca means "get current axes"
    # axex of plot, not image
    ax = plt.gca()

    # for each face, draw a rectangle based on coordinates
    for face in faces:
        x, y, width, height = face['box']
        face_border = Rectangle((x, y), width, height,
                          fill=False, color='red')
        ax.add_patch(face_border)
    # now show the plot with image+rectangle
    plt.show()


highlight_faces(image1, faces1)
highlight_faces(image2, faces2)


# extract face from full image
from numpy import asarray
def extract_face_from_image(image, faces,required_size=(224, 224)):
  # load image and detect faces

    face_images = []

    for face in faces:
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        # extract the face
        # numpy array
        face_boundary = image[y1:y2, x1:x2]

        # resize pixels to the model size
        # it's image,not array
        face_image = Image.fromarray(face_boundary)

        # resize image
        face_image = face_image.resize(required_size)

        # convert image to array
        face_array = asarray(face_image)

        # save array to list
        face_images.append(face_array)

    return face_images

# list of image arrays
extracted_faces1 = extract_face_from_image(image1, faces1)
extracted_faces2 = extract_face_from_image(image2, faces2)
print(extracted_faces1)

# Display the first face from the extracted faces
plt.imshow(extracted_faces1[0])
plt.show()

plt.imshow(extracted_faces2[0])
plt.show()


# comparing two faces
# pip3 install keras_vggface

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

# extracted_face is an array of image
def get_model_scores(extracted_faces):
    samples = asarray(extracted_faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object
    # this model saved as *.h5 at user->omi->.keras
    # image embedding
    model = VGGFace(model='resnet50',include_top=False,input_shape=(224, 224, 3),pooling='avg')

    print(model)

    # perform prediction
    # returns vector representation of images
    return model.predict(samples)


model_scores1 = get_model_scores(extracted_faces1)
print('model score: ')
print(model_scores1)

model_scores2 = get_model_scores(extracted_faces2)
print('model score: ')
print(model_scores1)


if cosine(model_scores1[0], model_scores2[0]) <= 0.4:
  print("Faces Matched")
else:
    print("Not Matched")

print(cosine(model_scores1[0], model_scores2[0]))
