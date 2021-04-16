import cv2,numpy,os
haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
detect = 'detect'
print("Training...")
(images,labels,names,id) = ([],[],{},0)    # id == no of datasets

for (subdirs,dirs,files) in os.walk(detect):
    for subdir in dirs:
        names[id]=subdir
        subjectPath = os.path.join(detect,subdir)
        for filename in os.listdir(subjectPath):
            path = subjectPath + '/' + filename
            label =id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1

(images,labels) = [numpy.array(lis) for lis in [images,labels]]
print(images,labels)
(width,heights) = (130,100)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images,labels)

cam = cv2.VideoCapture(0)
cnt=0

while True:
    ret,im = cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        face = gray[y:y+h,x:x+w]
        face_resize =cv2.resize(face,(width,heights))

        prediction = model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
        if prediction[1]<800:
            cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
            print(names[prediction[0]])
            cnt=0
        else:
            cnt+=1
            cv2.putTEXT(im,'Unknown',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            if cnt>100:
                print("Unknown Person")
                cv2.imwrite("Unknown.jpg",im)
                cnt=0
    cv2.imshow('Face Recognition',im)
    k=cv2.waitKey(1)
    if k==27:
        break
cam.release()
cv2.destroyAllWindows()
