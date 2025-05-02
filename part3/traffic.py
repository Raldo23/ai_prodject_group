import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_data(data_dir):
    images = []
    labels = []
    for label in range(43):
        folder = os.path.join(data_dir, str(label))
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (30, 30))
            images.append(img / 255.0)
            labels.append(label)
    return np.array(images), np.array(labels)


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(43, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python traffic.py gtsrb [model.h5]")
        return

    data_dir = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else "model.h5"

    print("Loading data...")
    X, y = load_data(data_dir)
    y = to_categorical(y, 43)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data loaded.")

    print("Training model...")
    model = build_model()
    history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=1)
    model.save(model_file)
    print(f"Model saved to {model_file}")

    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Model accuracy: {accuracy:.4f}")

    # Confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")

    # Visualize random predictions
    indices = np.random.choice(len(X_test), 5, replace=False)
    for i in indices:
        plt.figure()
        plt.imshow(X_test[i])
        plt.title(f"Predicted: {y_pred_classes[i]}, Actual: {y_test_classes[i]}")
        plt.savefig(f"prediction_{i}.png")
        plt.close()


if __name__ == "__main__":
    main()