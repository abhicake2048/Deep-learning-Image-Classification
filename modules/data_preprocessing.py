import tensorflow as tf
import keras

# Converting our data to a format which can be used for training.
def get_data(data_path,shuf=False):
    img_width = 256
    img_height = 256
    data = keras.utils.image_dataset_from_directory(
        data_path,
        shuffle=shuf,
        image_size = (img_width,img_height),
        batch_size = 32,
        validation_split=False
    )
    return data

