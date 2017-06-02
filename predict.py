from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import applications
import numpy as np

img_width, img_height = 150, 150
base_model=applications.VGG16(include_top=False,weights='imagenet')

def predict(basedir, targetclass, start, end):
    success = 0
    for i in range(start, end):
        path = basedir + str(i) + '.jpg'
        img = load_img(path,False,target_size=(img_width,img_height))
        img = img_to_array(img)
        img=img.reshape((1,)+img.shape)
        img=img/255
        feature_img =base_model.predict(img) #then the shape is (1,512,4,4)
        model=load_model('bottleneck_fc_model.h5') #my own model which the top is FC layers
        classes=model.predict_classes(feature_img)
        prob=model.predict_proba(feature_img)
        print("class: {0}".format(classes[0][0]))
        if classes[0][0] == targetclass:
        success = success + 1
    
    print(success)
    return success

start = 1401
end = 1411

basedir = "data/test/class1."
success_total = 0
success_total = success_total + predict(basedir, 0, start, end)

basedir = "data/test/class2."
success_total = success_total + predict(basedir, 1, start, end)

percent = success_total*100/(2*(end - start))

print('Result: {0}'.format(percent))
print('done')


