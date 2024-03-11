from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.optimizers import Adam


# Creating a sequential model and compiling it. 
def get_model(data_cat):
    model = Sequential()
    model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32,(3,3),1,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64,(3,3),1,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(len(data_cat),activation='softmax'))
    optimize = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimize,loss='SparseCategoricalCrossentropy',metrics = ['accuracy'])
    return model
