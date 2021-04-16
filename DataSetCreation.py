#pip install opencv-contrib-python
import cv2, os
haar_file = 'haarcascade_frontalface_default.xml'
detect= 'detect'
subData = 'Yuva'

path = os.path.join(detect,subData)
if not os.path.isdir(path):
    os.mkdir(path)
(width,height) = (130,100)

faceCascade = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(0)

count =1
while count<150:
    print(count)
    ret,frame = cam.read()
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        face = gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face,(width,height))
        cv2.imwrite('%s/%s.png' % (path,count),face_resize)
    count+=1

    cv2.imshow('Frame',frame)
    k = cv2.waitKey(1)
cam.release()
cv2.destroyAllWindows()