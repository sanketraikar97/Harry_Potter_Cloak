import cv2,numpy as np

def check(x):
    pass

# Create window and trackbars
cv2.namedWindow('Color Detector')

# Upper Limit HSV Trackbars
cv2.createTrackbar('Upper Hue','Color Detector',130,180,check)
cv2.createTrackbar('Upper Saturation','Color Detector',255,255,check)
cv2.createTrackbar('Upper Value','Color Detector',255,255,check)

# Lower Limit HSV Trackbars
cv2.createTrackbar('Lower Hue','Color Detector',65,180,check)
cv2.createTrackbar('Lower Saturation','Color Detector',30,255,check)
cv2.createTrackbar('Lower Value','Color Detector',60,255,check)

# Capture the intital for background creation
cap =cv2.VideoCapture(0)
while True:
    cv2.waitKey(2000)
    ret,intital = cap.read()
    if(ret):
        break

# Start capturing the frames 
while True:
    ret,frame = cap.read()
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Capture the Hue, Saturatrion and Value from Trackbars
    upper_hue = cv2.getTrackbarPos('Upper Hue','Color Detector')
    upper_sat = cv2.getTrackbarPos('Upper Saturation','Color Detector')
    upper_val = cv2.getTrackbarPos('Upper Value','Color Detector')
    lower_hue = cv2.getTrackbarPos('Lower Hue','Color Detector')
    lower_sat = cv2.getTrackbarPos('Lower Saturation','Color Detector')
    lower_val = cv2.getTrackbarPos('Lower Value','Color Detector')

    upper_limit = np.array([upper_hue,upper_sat,upper_val])
    lower_limit = np.array([lower_hue,lower_sat,lower_val])

    # Create a kernel for Dilation
    close_kernel = np.ones((9,9),np.uint8)
    open_kernel = np.ones((7,7),np.uint8)
    dilate_kernel = np.ones((10,10),np.uint8)

    # Create Mask
    mask = cv2.inRange(frame_hsv,lower_limit,upper_limit)
    mask = cv2.medianBlur(mask,3)
    # To get rid of the edges and get proper mask
    close_mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,close_kernel)
    open_mask = cv2.morphologyEx(close_mask,cv2.MORPH_OPEN,open_kernel)
    mask =cv2.dilate(open_mask,dilate_kernel,1)
    mask_inv = cv2.bitwise_not(mask)

    # Mix the frames
    b = frame[:,:,0]
    g = frame[:,:,1]
    r = frame[:,:,2]
    b = cv2.bitwise_and(mask_inv,b)
    g = cv2.bitwise_and(mask_inv,g)
    r = cv2.bitwise_and(mask_inv,r)
    frame_inv = cv2.merge((b,g,r))

    b = intital[:,:,0]
    g = intital[:,:,1]
    r = intital[:,:,2]
    b = cv2.bitwise_and(b,mask)
    g = cv2.bitwise_and(g,mask)
    r = cv2.bitwise_and(r,mask)
    cloak = cv2.merge((b,g,r))

    final = cv2.bitwise_or(frame_inv,cloak)

    cv2.imshow('cloak',final)
    cv2.imshow('Origianl',frame)

    if cv2.waitKey(5) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()