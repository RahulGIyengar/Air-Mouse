import numpy as np
import Track as htm
import time
import autopy

wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7 
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
 
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)
 
while True:
    #hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    
    #Finding tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList&#91;8]&#91;1:]
        x2, y2 = lmList&#91;12]&#91;1:]
        # print(x1, y1, x2, y2)
    
    #Checking which finger is pointing UP
    fingers = detector.fingersUp()
    # print(fingers)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
    (255, 0, 255), 2)
    
    #Move using only Index finger
    if fingers&#91;1] == 1 and fingers&#91;2] == 0:
    
        #Convert Coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
        
        #Smoothen
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
    
        #Move cursor
        autopy.mouse.move(wScr - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY
        
    #Click using both fingers
    if fingers&#91;1] == 1 and fingers&#91;2] == 1:
    
        #Linear distance between fingers
        length, img, lineInfo = detector.findDistance(8, 12, img)
        print(length)
        
        #Short linear distance enables click
        if length &lt; 40:
            cv2.circle(img, (lineInfo&#91;4], lineInfo&#91;5]),
            15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()
    
    #Calculate frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    (255, 0, 0), 3)
    
    #Display results
    cv2.imshow("Image", img)
    cv2.waitKey(1)
