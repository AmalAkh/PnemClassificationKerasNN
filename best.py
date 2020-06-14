import numpy as np
import cv2
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import keras.utils as ut
import os
from keras.regularizers import l2

def load_images_to_data(image_directory, features_data, label_data):
    list_of_dirs = os.listdir(image_directory)
    classes = []
    for dir in list_of_dirs:
        if dir != ".DS_Store":
            print(dir)
            classes.append(dir)

            list_of_files = os.listdir(image_directory + dir)

            for file in list_of_files:

                if file != ".DS_Store":
                    image_file_name = image_directory + dir + "/" + file

                    img = cv2.imread(image_file_name)
                    # img = np.resize(img, (28,28,1))
                    print(image_file_name)
                    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.array(img)

                    features_data.append(img)
                    label_data.append(classes.index(dir))

    print(classes)
    return features_data, label_data


x_train = []
y_train = []
x_test = []
y_test = []
x_val = []
y_val = []
x_train, y_train = load_images_to_data("chest_xray/train/", x_train, y_train)
x_test, y_test = load_images_to_data("chest_xray/test/", x_test, y_test)
x_val, y_val = load_images_to_data("chest_xray/val/", x_val, y_val )



x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train /255
x_test = x_test /255

y_train = ut.to_categorical(y_train, 2)
y_test = ut.to_categorical(y_test, 2)


model = Sequential()



model.add(Conv2D(128, kernel_size=3, activation="relu", kernel_regularizer=l2(0.0001)))
#model.add(Conv2D(256, kernel_size=3, activation="relu",kernel_regularizer=l2(0.0001)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))


model.add(Conv2D(128, kernel_size=3, activation="relu",kernel_regularizer=l2(0.0001)))
model.add(Conv2D(128, kernel_size=3, activation="relu",kernel_regularizer=l2(0.0001)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=3, activation="relu",kernel_regularizer=l2(0.0001)))
model.add(Conv2D(32, kernel_size=3, activation="relu",kernel_regularizer=l2(0.0001)))

model.add(Dropout(0.2))




model.add(Flatten())
model.add(Dense(512, activation="relu",kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.2))

model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.2))


model.add(Dense(2, activation="softmax",kernel_regularizer=l2(0.0001)))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



model.fit(x_train, y_train, epochs=2, batch_size=32, validation_split=0.05, callbacks=[EarlyStopping(monitor="val_loss", patience=5)])

res = model.evaluate(x_test, y_test)
print(res)
model.save("best-model.h5")
