import cv2, sys, numpy, os
size = 2
frontal_face_haar = 'haarcascade_frontalface_default.xml'
faces_database_path = 'imgs'

(images, lables, names, id) = ([], [], {}, 0)
for (image_paths, apaths, files) in os.walk(faces_database_path):
    
    for image_path in apaths:
        names[id] = image_path
        subjectpath = os.path.join(faces_database_path, image_path)
        for filename in os.listdir(subjectpath):
            f_name, f_extension = os.path.splitext(filename)
            if(f_extension.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']):
                #print("Skipping this file called  "+filename+", because it is a wrong file type")
                continue
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(im_width, im_height) = (100, 100)
(images, lables) = [numpy.array(lis) for lis in [images, lables]]
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, lables)
haar_like_feature = cv2.CascadeClassifier(frontal_face_haar)
open_webcam = cv2.VideoCapture(0)
while True:
    check_webcam = False
    while(not check_webcam):
        (check_webcam, frame) = open_webcam.read()
        if(not check_webcam):
            print("Problem with the webcam")
    frame=cv2.flip(frame,1,0)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mini_image = cv2.resize(grayscale, (int(grayscale.shape[1] / size), int(grayscale.shape[0] / size)))
    faces = haar_like_feature.detectMultiScale(mini_image)
    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * size for v in face_i]
        face = grayscale[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        prediction = model.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print(prediction)
        print(names[prediction[0]])
        if(prediction[1]>200):
            cv2.putText(frame,'Unknown',(x-5, y-5), cv2.FONT_ITALIC,0.5,(0,0,255))
        else:
            cv2.putText(frame,'%s' % (names[prediction[0]]),(x-5, y-5), cv2.FONT_ITALIC,0.5,(255,0,0))
    cv2.imshow('RGUKT Student Tracking System', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
