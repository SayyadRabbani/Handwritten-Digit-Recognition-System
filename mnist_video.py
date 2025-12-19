import tensorflow as tf
import numpy as np
import cv2

# -------------------------------------------------
# Load MNIST data
# -------------------------------------------------
def get_mnist_data():
    path = 'mnist.npz'
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    return x_train, y_train, x_test, y_test


# -------------------------------------------------
# Train MNIST model
# -------------------------------------------------
def train_model(x_train, y_train, x_test, y_test):

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') > 0.99:
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        callbacks=[myCallback()]
    )

    print("Final accuracy:", history.history['accuracy'][-1])
    return model


# -------------------------------------------------
# Predict digit
# -------------------------------------------------
def predict(model, img):
    img = img / 255.0
    img = np.array([img])
    prediction = model.predict(img, verbose=0)
    return str(np.argmax(prediction))


# -------------------------------------------------
# OpenCV mouse callback
# -------------------------------------------------
startInference = False

def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference


# -------------------------------------------------
# Threshold trackbar callback
# -------------------------------------------------
threshold = 150

def on_threshold(val):
    global threshold
    threshold = val


# -------------------------------------------------
# OpenCV loop
# -------------------------------------------------
def start_cv(model):
    global threshold

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)
    cv2.createTrackbar('threshold', 'background', threshold, 255, on_threshold)

    background = np.zeros((480, 640), np.uint8)
    frameCount = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if startInference:
            frameCount += 1

            frame[0:480, 0:80] = 0
            frame[0:480, 560:640] = 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

            roi = thr[240-75:240+75, 320-75:320+75]
            background[240-75:240+75, 320-75:320+75] = roi

            digit_img = cv2.resize(roi, (28, 28))
            result = predict(model, digit_img)

            if frameCount == 5:
                background[:] = 0
                frameCount = 0

            cv2.putText(
                background, result, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3
            )

            cv2.rectangle(
                background,
                (320-80, 240-80),
                (320+80, 240+80),
                255, 3
            )

            cv2.imshow('background', background)

        else:
            cv2.imshow('background', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    try:
        model = tf.keras.models.load_model('model.keras')
        print("Loaded saved model.")
        print(model.summary())
    except:
        print("Getting MNIST data...")
        x_train, y_train, x_test, y_test = get_mnist_data()
        print("Training model...")
        model = train_model(x_train, y_train, x_test, y_test)
        print("Saving model...")
        model.save('model.keras')

    print("Starting webcam...")
    start_cv(model)


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == '__main__':
    main()
