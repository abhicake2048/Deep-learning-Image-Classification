from modules import model as md
from modules import data_preprocessing as dp
from modules import predict 
import numpy as np

# Specifying the path to our data folders 
data_train_path='data/train'
data_test_path='data/test'
data_val_path='data/validation'

# Getting pre-processed data
train_data = dp.get_data(data_train_path,True)
test_data = dp.get_data(data_test_path)
val_data = dp.get_data(data_val_path)

# Getting the class names
data_cat = train_data.class_names

# scaling the data
train_data = train_data.map(lambda x,y :(x/255,y))
test_data = test_data.map(lambda x,y :(x/255,y))
val_data = val_data.map(lambda x,y :(x/255,y))

# creating the model
model = md.get_model(data_cat)

# Training the model
cycle = 1
history = model.fit(train_data,validation_data=val_data,epochs=cycle,batch_size=32,verbose=1)

model.save("image.keras")

image = 'carrot.jpg'

pred,scor = predict.get_predict(image,model)

ans = data_cat[np.argmax(scor)]

print('The image is of a {}'.format(ans))
