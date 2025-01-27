import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Haar-Cascade Classifier
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function for detecting the face in an image
def detect_face(img):
    face = detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5) # tune if needed

    if len(face)>=1:
        return face[0] # only detect the first and most prominent face
    else:
        return None

# Function for cropping the face in the image
def crop_face(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
    face_rect = detect_face(img_gray) if img_gray is not None else None

    if face_rect is not None:
        x, y, w, h = face_rect
        img_cropped = img_gray[y:y+h, x:x+w]
        return img_cropped
    else:
        return None

def menu_1(train_path):
    train_images = []
    train_labels = []
    name_list = []

    print("Collecting data...")
    for label, person_folder in enumerate(os.listdir(train_path)):
        name_list.append(person_folder)
        person_path = os.path.join(train_path, person_folder)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            cropped_img = crop_face(img)
            if cropped_img is not None:
                train_images.append(cropped_img)
                train_labels.append(label)

    x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

    # Dataset Information
    print(f"\nTrain: Got {len(x_train)} images with total of {len(y_train)} labels")
    print(f"Validation: Got {len(x_val)} images with total of {len(y_val)} labels")
    print(f"Names (Index determine the Label): {name_list}\n")

    # Create and train the face recognizer
    print("Training model...")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=4, neighbors=8) # tune if needed
    face_recognizer.train(x_train, np.array(y_train))

    # Make predictions on the test data
    print("Evaluating model...")
    correct = 0
    total = len(x_val)
    for idx, val_image in enumerate(x_val):
        prediction, distance = face_recognizer.predict(val_image)
        actual = y_val[idx]
        print(f"Actual: {name_list[actual]}, Predicted: {name_list[prediction]}, Confidence: {distance:.2f}")
        if prediction==actual:
            correct += 1

    # Calculate accuracy
    accuracy = (correct / total) * 100
    print(f"Total Prediction: {total}")
    print(f"Correct Prediction: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Save the model and the name list
    face_recognizer.write("./fr_model.yml")
    np.save("./name_list.npy", name_list)
    print("\nModel and names saved successfully!\n")

def menu_2(image_path):
    try:
        # Load the pre-trained model and name list
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read("./fr_model.yml")
        name_list = np.load("./name_list.npy")
    except:
        print("\nError: Model or name list file not found! Please choose menu 1 first!\n")
        return

    # Check if path exists
    if not os.path.exists(image_path):
        print("\nPath doesn't exist!\n")
        return
    
    # Check if the file is indeed an image file
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("\nNo image in file!\n")
        return
    
    cropped_img = crop_face(img_bgr)

    if cropped_img is not None:
        res, distance = face_recognizer.predict(cropped_img)

        face_rect = detect_face(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
        x, y, w, h = face_rect

        name = name_list[res]
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img_bgr, f"{name} : {distance:.2f}", (x, y-10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("\nPrediction Completed!\n")

    else:
        print("\nNo face detected!\n")
        return

def main_menu():
    while True:
        print("Menu")
        print("1. Train and Test Model")
        print("2. Test Recognition")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            train_path = "./images/train/" # path to train folder
            menu_1(train_path)

        elif choice == '2':
            path = input("Enter the path to the image for testing (Absolute Path): ")
            menu_2(path)

        elif choice == '3':
            print("\nProgram Exited!\n")
            break

        else:
            print("\nInvalid Choice!\n")

if __name__ == "__main__":
    main_menu()