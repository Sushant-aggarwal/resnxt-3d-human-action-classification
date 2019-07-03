import os
import numpy as np
import cv2
from random import shuffle
from tqdm import tqdm

IMG_SIZE = 112
TRAIN_DIR = 'train/'
TEST_DIR = 'test/'

def label_img(img):
	cl = img.split('.')[1]
	x = np.zeros((1,3))
	x[0][int(cl)-1] = 1
	return x

def create_train_data():
	train_X = []
	train_Y = []
	for d in tqdm(os.listdir(TRAIN_DIR)):
		X = []
		label = label_img(d)
		path = os.path.join(TRAIN_DIR,d)
		for video in tqdm(os.listdir(path)):
			count=0
			X=[]
			vid=cv2.VideoCapture(os.path.join(path,video)) 
			length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
			i=0
			k=length/5
			k=length/16
			k=int(k)
			j=0
			framediff=5
			count2=0
			while True:
				i+=1
				
				if(count2==k):
					break
				ret,frame= vid.read()
				if not ret:
					break
			
        		
				if(i==(1+(j*framediff))):
					frame=cv2.resize(frame,(112,112))
					X.append(np.array(frame))
					count+=1
					j+=1
			
				if(count==16):	
					train_X.append(np.array(X))
					train_Y.append(np.array(label))
					count=0
					count2+=1
					
		

	train_X = np.array(train_X)
	train_Y =np.array(train_Y)
	np.save('train_X.npy',train_X)
	np.save('train_Y.npy',train_Y)

	
def create_test_data():
	test_X = []
	test_Y = []
	for d in tqdm(os.listdir(TEST_DIR)):
		X = []
		label = label_img(d)
		path = os.path.join(TEST_DIR,d)
	for video in tqdm(os.listdir(path)):
			count=0
			X=[]
			vid=cv2.VideoCapture(os.path.join(path,video)) 
			length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
			i=0
			k=length/5
			k=length/16
			k=int(k)
			j=0
			framediff=5
			count2=0
			while True:
				i+=1
				
				if(count2==k):
					break
				ret,frame= vid.read()
				if not ret:
					break
			
        		
				if(i==(1+(j*framediff))):
					frame=cv2.resize(frame,(112,112))
					X.append(np.array(frame))
					count+=1
					j+=1
			
				if(count==16):	
					test_X.append(np.array(X))
					test_Y.append(np.array(label))
					count=0
					count2+=1
					
	test_X = np.array(test_X)
	test_Y =np.array(test_Y) 
	np.save('test_X.npy',test_X)
	np.save('test_Y.npy',test_Y)

create_train_data()
create_test_data()
