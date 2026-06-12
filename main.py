import numpy as np
import os
import cv2
import h5py
from sklearn.model_selection import train_test_split


# =========================
# LOAD DATASET
# =========================
def load_images(root_dir, img_size=64):
    X = []
    Y = []

    categories = ["Day", "Night"]
    label_map = {"Day": 0, "Night": 1}

    for split in ["train", "test"]:
        for category in categories:
            folder_path = os.path.join(root_dir, split, category)

            if not os.path.exists(folder_path):
                continue

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)

                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.resize(img, (img_size, img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                X.append(img)
                Y.append(label_map[category])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


# =========================
# CREATE HDF5 FILE
# =========================
def create_h5(X_train, Y_train, X_test, Y_test, filename="data.h5"):
    with h5py.File(filename, "w") as f:
        f.create_dataset("train_set_x", data=X_train)
        f.create_dataset("train_set_y", data=Y_train)
        f.create_dataset("test_set_x", data=X_test)
        f.create_dataset("test_set_y", data=Y_test)
    print("data.h5 created successfully!")


# =========================
# SIGMOID
# =========================
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# =========================
# INIT PARAMETERS
# =========================
def initialize(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


# =========================
# FORWARD + BACKWARD
# =========================
def propagate(w, b, X, Y):
    m = X.shape[1]

    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    A = np.clip(A, 1e-8, 1 - 1e-8)

    cost = -(1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)

    return dw, db, cost


# =========================
# OPTIMIZATION
# =========================
def optimize(w, b, X, Y, iterations=2000, lr=0.001):
    costs = []

    for i in range(iterations):
        dw, db, cost = propagate(w, b, X, Y)

        w = w - lr * dw
        b = b - lr * db

        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after {i}: {cost}")

    return w, b, costs


# =========================
# PREDICTION
# =========================
def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)


# =========================
# MODEL
# =========================
def model(X_train, Y_train, X_test, Y_test):
    w, b = initialize(X_train.shape[0])

    w, b, costs = optimize(w, b, X_train, Y_train)

    Y_pred_train = predict(w, b, X_train)
    Y_pred_test = predict(w, b, X_test)

    print("\nFinal Results:")
    print("Train accuracy:", 100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100)
    print("Test accuracy:", 100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100)

    return w, b


# =========================
# RUN PIPELINE
# =========================
if __name__ == "__main__":

    X, Y = load_images("Datasets", img_size=64)

    print("Loaded X shape:", X.shape)
    print("Loaded Y shape:", Y.shape)

    X = X / 255.0
    X = X.reshape(X.shape[0], -1).T
    Y = Y.reshape(1, -1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X.T, Y.T, test_size=0.2, random_state=42
    )

    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.T
    Y_test = Y_test.T

    # CREATE HDF5 FILE (IMPORTANT FOR ASSIGNMENT)
    create_h5(X_train, Y_train, X_test, Y_test)

    # TRAIN MODEL
    w, b = model(X_train, Y_train, X_test, Y_test)

 