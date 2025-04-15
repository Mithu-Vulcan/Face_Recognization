from deepface import DeepFace
import cv2
import threading


#Capture
cap =cv2.VideoCapture(0)

#Capture setting size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4800)

#count of the times where we got the return_value from cv.read()
#so that we can control when we check the face
#down inside the mainloop's if statement we used it.
counter = 0

#check if the face matches or not
face_match =False

#image to compare with
reference_image = cv2.imread("Photo.png")

#function to check the face
def check_face(frame):
    global face_match
    #the verified is true only when the 2 frames matchs
    try:
        if DeepFace.verify(frame, reference_image.copy())['verified']:
            face_match = True
        else:
            face_match = False

    except ValueError:
        face_match = False

#main loop
while True:

    #get the two outputs of reading the capture
    return_val, frame = cap.read()

    #to check if there was a return value
    if return_val:
        #to only start the thread every 30 frames that we get
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        #checks if there is a frame
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3 )
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3 )

        cv2.imshow("Video", frame)

    #break the loop on key press
    key = cv2.waitKey(1)
    if key == ord("q"):
        break


cv2.destroyAllWindows()