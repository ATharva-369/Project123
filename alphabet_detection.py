import cv2
import numpy as np
import pandas as pd 
import seaborn as sb 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os , ssl , time

#loading the data using numpy and pandas
X = np.load('image.npz')['arr_0']
Y = pd.read_csv("labels.csv")["labels"]


#creating the classes for all the alphabets
classes = ['A', 'B', 'C' ,'D', 'E', 'F', 'G' ,'H' ,'I' ,'J' ,'K' ,'L', 'M', 'N' ,'O', 'P', 'Q','R', 'S', 'T', 'U' ,'V' ,'W' ,'X', 'Y' ,'Z']
nclasses = len(classes) #storing the length of the number of alphabets
samples = 3
fig = plt.figure(figsize = (nclasses*2,(1+samples*2))) # plotting a figure and setting the figssize according to the list's size
index = 0

#spliting and scaling the testing and training data
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 2500,random_state = 9,train_size=7500) 
x_trainscale = x_train / 255.0 
x_testscale = x_test / 255.0

#creating a Logistic Regression model and fitting our data
clf = LogisticRegression(solver = 'saga', multi_class='multinomial')
clf.fit(x_trainscale,y_train)
#creating a prediction model and printing the accuracy of the data
predict = clf.predict(x_testscale)
accuracy = accuracy_score(y_test,predict)
print(f'the accuracy is {accuracy*100}%')

#starting the camera
cap = cv2.VideoCapture(0)

#creating a while loop to capture frames
while(True):
    try :
        ret,frame = cap.read()

        #changing the frame to gray
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #drawing a rectangle to focus on our image:
        height,width = gray.shape
        upper_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right = (int(width/2 + 56), int(height/2+56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)

        #we will only consier the area inside the reectangle
        #eoi = Region Of Interest
        roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]

        #convert the image into a pil format
        im_pil = Image.fromarray(roi)

        #convert to grayscale image
        image_bw = im+pil.convert('L')
        image_bw_resized = image_bw_resize((28,26), Image.ANTIALIAS)

        #invert the image
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20

        #converting to scalar
        min_pixel = np.percenile(image_bw_resized_inverted,pixel_filter)

        #using clip to limit the values between 0 and 250
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel,0,255)
        max_pixel = np.max(image_bw_resized_inverted)

        #converting into an array

        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        #creating a test sample and creating a prediction
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf,predict(test_sample)
        print("the predicted alphabet is: ",test_pred)

        #display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
            pass
cap.release()
cv2.distroyAllWindows()        

