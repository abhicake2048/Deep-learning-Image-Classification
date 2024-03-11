import keras
import tensorflow as tf

# making the prediction function.
def get_predict(image_path,model):
    img_width = 256
    img_height = 256
    image = image_path
    image = keras.utils.load_img(image,target_size=(img_height,img_width))
    img_arr=keras.utils.array_to_img(image)
    img_bat = tf.expand_dims(img_arr,0)
    pred = model.predict(img_bat)
    scor = tf.nn.softmax(pred)
    return pred,scor