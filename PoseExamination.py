import cv2
import mediapipe as mp
import time
import resize

class poseDectector():
    def __init__(self,mode= False,model_complexity= 1,smooth_landmarks = True,
                 enable_segmentation = False,smooth_segmentation = True,
                 min_detection_confidence = 0.5,min_tracking_confidence = 0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity,
                                     self.smooth_landmarks, self.enable_segmentation,
                                     self.smooth_segmentation, self.min_detection_confidence,
                                     self.min_tracking_confidence)

    def findPose(self, img, draw= True):

        img = resize.resizeframe(img)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB)
        #print(result.pose_landmarks)
        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw= True):
        lmList = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture('Video/pexels-julia-larson-6454280.mp4')
    pTime = 0
    dectector = poseDectector()

    while True:
        success, img = cap.read()
        img = dectector.findPose(img)
        lmList = dectector.findPosition(img)
        #print(lmList)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        if len(lmList) !=0:
            print(lmList)
            cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,
                        (255,0,0),3)

        cv2.imshow("img", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()