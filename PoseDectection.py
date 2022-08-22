import cv2
import time
import PoseExamination as pe


cap = cv2.VideoCapture('Video/pexels-julia-larson-6454280.mp4')
pTime = 0
dectector = pe.poseDectector()

while True:
    success, img = cap.read()
    img = dectector.findPose(img)
    lmList = dectector.findPosition(img)
    #print(lmList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    if len(lmList) != 0:
        print(lmList)
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

    cv2.imshow("img", img)
    cv2.waitKey(1)
