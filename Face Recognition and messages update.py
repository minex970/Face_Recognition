#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## ALL modules...
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import time

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import getpass

import pywhatkit as kit

import subprocess as s


# In[ ]:


# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[ ]:


def email(): 
    passwd = getpass.getpass("Enter the Sender Email password :")
    fromaddr = "test.project970@gmail.com"
    toaddr = "gneeraj970@gmail.com"

    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Face Reognized"
    body = "Hurray Your Face Recognition code working !!!!"
    msg.attach(MIMEText(body, 'plain'))

    filename = "neeraj.jpg"
    attachment = open(r"./neeraj.jpg", "rb")

    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(p)
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(fromaddr,passwd)
    text = msg.as_string()
    s.sendmail(fromaddr,toaddr,text)
    s.quit()
    print("\nMail sent....check Inbox !!!!\n")


# In[ ]:


def whatsapp():
    phone = input("Enter your's friend number: ")
    kit.sendwhatmsg_instantly("+91"+phone,"Security Alert : Someone Face has been Detected" ,20 ,False)
    print("whatsapp msg sent !!!!\n")


# In[ ]:


def aws():
    
#Launching EC2 Instance
    instances = input("Enter instances type: ")
    value = input("Enter the name for your instances: ")
    print("launching instances...\napproximately it's take 2-3 mins to launch.")
    
    output1 = s.getstatusoutput("aws ec2 run-instances --image-id ami-011c99152163a87ae  --count 1 --instance-type {} --key-name task6 --security-group-ids sg-07108f193958782d9 --subnet-id subnet-5dede435".format(instances))
    instance_id = output1[1][157:176]
    tmp = s.getstatusoutput("aws ec2 create-tags --resources {} --tags Key=Name,Value={}".format(instance_id,value))
    time.sleep(180)
    print("Instances created.\n")

#Creating EBS volume
    
    size = input("Enter the size of EBS volume(in GB): ")
    name = input("Enter the name for your volume:")
    print("Creating EBS volume...")
    output2 = s.getstatusoutput("aws ec2 create-volume --volume-type gp2  --size {} --availability-zone ap-south-1a".format(size))
    vol_id = output2[1][191:212]
    tmp = s.getstatusoutput("aws ec2 create-tags --resources {} --tags Key=Name,Value={}".format(vol_id,name))
    print("Volume created.\n")

#Attaching the EBS Volume to the EC2 Instance
    print("Attaching the EBS volume...")
    tmp = s.getstatusoutput("aws ec2 attach-volume --volume-id {} --instance-id {} --device /dev/sdf".format(vol_id,instance_id))
    print("Successfully attach EBS volume to instances.")


# In[ ]:


## Collecting Images

# Load functions
def face_extractor(img):
   # Function detects faces and returns the cropped face
   # If no face detected, it returns the input image
   
   gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   faces = face_classifier.detectMultiScale(gray, 1.3, 5)
   
   if faces is ():
       return None
   
   # Crop all faces found
   for (x,y,w,h) in faces:
       cropped_face = img[y:y+h, x:x+w]
   return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
print("collecting 100 samples of your face...")
while True:

   ret, frame = cap.read()
   if face_extractor(frame) is not None :
       count += 1
       face = cv2.resize(face_extractor(frame), (200, 200))
       face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

       # Save file in specified directory with unique name
       file_name_path = './faces/' + 'grp4-'  + str(count) + '.jpg'
       cv2.imwrite(file_name_path, face)

       # Put count on images and display live count
       cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
       cv2.imshow('Face Cropped', face)
       print("Faces sample {}".format(count))        
   else:
       print("Face not found")
       pass

   if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
       break
       
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")


# In[ ]:


## Training Model

# Get the training data we previously made
data_path = './faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
   image_path = data_path + onlyfiles[i]
   images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
   Training_Data.append(np.asarray(images, dtype=np.uint8))
   Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

model  = cv2.face_LBPHFaceRecognizer.create()
# Let's train our model 
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")


# ## 6.1- If the Face Recognise then send email to us and WhatsApp to your friend..

# In[ ]:


## Face Recognise

def face_detector(img, size=0.5):    
   # Convert image to grayscale
   gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   faces = face_classifier.detectMultiScale(gray, 1.3, 5)
   if faces is ():
       return img, []
      
   for (x,y,w,h) in faces:
       cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
       roi = img[y:y+h, x:x+w]
       roi = cv2.resize(roi, (200, 200))
   return img, roi

# Open Webcam
cap = cv2.VideoCapture(0)
while True:

   ret, frame = cap.read()    
   image, face = face_detector(frame)    
   try:
       face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
       # Pass face to prediction model
       # "results" comprises of a tuple containing the label and the confidence value
       results = model.predict(face)
       # harry_model.predict(face)
       
       if results[1] < 500:
           confidence = int( 100 * (1 - (results[1])/400) )
           display_string = str(confidence) + '% Confident it is User'
           
       cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
       
       if confidence > 75:
           cv2.putText(image, "Neeraj's photo", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
           cv2.imshow('Face Recognition', image)
           cv2.imwrite("neeraj(6.1).jpg" , image)
           email()
           whatsapp() 
           break        
       else:     
           cv2.putText(image, "U R FRIEND!!!, how r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
           cv2.imshow('Face Recognition', image )

   except:
       cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
       cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
       cv2.imshow('Face Recognition', image )
       pass
       
   if cv2.waitKey(1) == 13: #13 is the Enter Key
       break
       
cap.release()
cv2.destroyAllWindows()
print("ALL set...!!!\nDone with Task-6.1")


# ## 6.2- If the Face Recognise then launch EC2 instance, EBS volume and attach it to the instance..

# In[ ]:


## Face Recognise

def face_detector(img, size=0.5):   
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()    
    image, face = face_detector(frame)    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = model.predict(face)
        # harry_model.predict(face)
        
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 85:
            cv2.putText(image, "Neeraj's second photo", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image)
            cv2.imwrite("neeraj(6.2).jpg" , image)
            aws()
            break         
        else:            
            cv2.putText(image, "U R FRIEND!!!, how r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()
print("ALL set...!!!\nDone with Task-6.2")


# In[ ]:




