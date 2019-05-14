import cv2
import numpy as np
import os

train_data_loc = 'D:\\ml and dl\\DeepLearningAlgos\\NNCode\\dogs_cats\\train'
validation_data_loc = 'D:\\ml and dl\\DeepLearningAlgos\\NNCode\\dogs_cats\\validation_sample'
test_data_loc = 'D:\\ml and dl\\DeepLearningAlgos\\NNCode\\dogs_cats\\test'

def img_label(img_path):
    word_label = img_path.split('.')[0]
    if word_label == 'cat': return 0
    else: return 1

def resize(img):
    ratio = 70/img.shape[0] #new height/old height
    dim = (int(img.shape[1]*ratio),70) #width*ratio
    img = cv2.resize(img,dim)
    return img

def create_data(flag):
    dataset = []
    data_loc = None
    stored_file = None
    if flag == 'validation':
        data_loc = validation_data_loc
        stored_file = 'validation_data.npy'
    elif flag == 'train':
        data_loc = train_data_loc
        stored_file = 'train_data.npy'
    elif flag == 'test':
        data_loc = test_data_loc
        stored_file = 'test_data.npy'
    for img_name in os.listdir(data_loc):
        img_path = os.path.join(data_loc,img_name)
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img,dsize=(60,60))
        if flag == 'train' or flag == 'validation':
            label = img_label(img_name)
            dataset.append([gray_img,label])
        elif flag == 'test':
            dataset.append(gray_img)
    np.random.shuffle(dataset)
    np.save(stored_file,dataset)

create_data('train')
create_data('test')
#create_data('validation')