# Facial-Recognition-Attendance-System
A Facial Recognition Attendance System developed using Python and OpenCV. This project automatically marks attendance by detecting and recognizing faces, reducing manual effort and improving accuracy. It includes features for adding new users, storing attendance records, and differentiating between recognized and unrecognized individuals.
## ✨ Features
- Real-time face detection and recognition using **OpenCV**  
- Automatic attendance marking for recognized users  
- Differentiates between registered and unregistered individuals  
- Stores attendance in **CSV/Excel/Database**  
- Easy registration of new users  
- User-friendly interface for running the system  

---

## 🛠️ Technologies Used
- **Python 3**  
- **OpenCV**  
- **NumPy**  
- **Pandas**  
- **SQLite / CSV / Excel**  

Facial-Recognition-Attendance-System/
│
├── main.py                 # Main file to run the system
├── train.py                # Script to train the recognition model
├── requirements.txt        # Required dependencies
├── dataset/                # Stores facial images of users
├── models/                 # Trained recognition model files
├── attendance.csv          # Attendance records
└── README.md               # Project documentation

## Code 
import cv2
import face_recognition
 
imgElon = face_recognition.load_image_file('ImagesBasic/Mohsin.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Faizan.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
 
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
 
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKe

