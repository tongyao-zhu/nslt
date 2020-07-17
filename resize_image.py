#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json


import torchvision.transforms as transforms
from PIL import Image
 
from torch import nn
import os
from multiprocessing import Pool


CORPUS_PATH = "./../generate-googLeNet-features/old_corpus/"
NUM_PROCESS = 12
# In[3]:


def get_image_paths(video_path):
    image_names= sorted(filter(lambda x: x[-4:]==".png", os.listdir(video_path)))
    
    full_image_paths = list(map(lambda x: (video_path +x ).replace("\n",""), image_names))
    images_list = full_image_paths
    return images_list


# In[5]:


# processing corpus
image_name_list = []
for mode in ["dev","test","train"]:
    print("processing {} corpus".format(mode))
    old_dev_corpus = open(os.path.join(CORPUS_PATH, "phoenix2014T.{}.sign".format(mode)), "r").readlines()
    
    count = 0
    for folder_path in old_dev_corpus:
        count+=1
        if not count%(len(old_dev_corpus)//10):
            print("currently processed {} of {} ({}%)".format(count,len(old_dev_corpus),count*100//len(old_dev_corpus)))
        real_path = (folder_path.replace("<PATH_TO_EXTRACTED_AND_RESIZED_FRAMES>", "./../PHOENIX-2014-T-release-v3/PHOENIX-2014-T")).replace("227x227","210x260").replace("\n","")
        image_name_list.extend(get_image_paths( real_path))
    print('after the current section, count is {}'.format(count))


# In[ ]:

def resize_image(image_name):
    image = Image.open(image_name)

    new_image = image.resize((227, 227))
    folder_name = image_name.replace("210x260","227x227")[:-14]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    new_image.save(image_name.replace("210x260","227x227"))

pool = Pool(NUM_PROCESS)
pool.map(resize_image, image_name_list)
pool.close()
pool.join()
print("Finished process!")



