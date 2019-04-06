from keras.models import load_model
import cv2 as cv
import argparse
import numpy as np

emotions = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]

def predict(model_file, image_file):
    faces = process_image(image_file)
    model = load_model(model_file)
    predictions = model.predict(faces)
    print("Predictions probabilities:", predictions)
    prediction_labels = [emotions[np.argmax(predictions[i])] for i in range(len(predictions))]
    return prediction_labels

def process_image(image_file):
    face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

    image_gray = cv.cvtColor(cv.imread(image_file), cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.1, 6)
    print ("Found {0} faces in image".format(len(faces)))

    processed_faces = []
    for (x, y, w, h) in faces:
        cropped = image_gray[y : y+h, x : x+w]
        resized = cv.resize(cropped,(48,48))
        scaled = resized.reshape(48,48,1) / 255
        processed_faces.append(scaled)
    
    return np.array(processed_faces)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="trained model file")
    parser.add_argument("image", help="image file to predict")
    args = parser.parse_args()

    np.set_printoptions(suppress=True)

    predictions = predict(args.model, args.image)
    print("Predicted emotions: ", predictions)