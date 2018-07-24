import numpy as np
import cv2

cap = cv2.VideoCapture('2.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    
    mask_rgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
    out_frame = cv2.bitwise_and(frame, mask_rgb)
    #test_out_frame = cv2.bitwise_and(frame, frame, mask=fgmask)
    #cv2.imshow("F1G", test_out_frame)
    cv2.imshow("FG", out_frame)
 
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()


























