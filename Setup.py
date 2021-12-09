import urllib.request
import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO
from IPython.display import Audio
import pyttsx3
global k

def process_image(img):
    """Resize, reduce and expand image.
    # Argument:
        img: original image.
    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)
    return image 

def get_classes(file):
    """Get classes name.
    # Argument:
        file: classes name for database.
    # Returns
        class_names: List, classes name.
    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

# distance from camera to object(face) measured in cm
Known_distance = 76.2
# width of face in the real world or Object Plane in cm
Known_width = 14.3
#focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
	# finding the focal length
	focal_length = (width_in_rf_image * measured_distance) / real_width
	return focal_length
# distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
	distance = (real_face_width * Focal_Length)/face_width_in_frame
	# return the distance
	return distance
# reading reference_image from directory
ref_image = cv2.imread("Ref.png")

def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.
    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, 0.318306595087051)

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
        Distance = Distance_finder(Focal_length_found, Known_width, w)*1000
        # defining the rules and finding the direction
        pos_h=x+w/2
        pos_v=y+h/2
        if pos_h<192 and pos_h>=0:
            direction= "left"
        elif pos_h>=192 and pos_h<=448:
            if pos_v<384 and pos_v>=0:
                direction= "front"
            elif pos_v>=384 and pos_v<480:
                direction="bottom"
        elif pos_h>448 and pos_h<640:
            direction="right"

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f} {2} {3}'.format(all_classes[cl], score, Distance, direction),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)
        if Distance<=200 and Distance>=0:
            # initialize Text-to-speech engine  
            engine = pyttsx3.init()  
            # convert this text to speech  
            ndis= int(Distance)
            text = all_classes[cl] + " is at "+ str(ndis) + "centimeter at the"+ direction
            engine.say(text)  
            # play the speech  
            engine.runAndWait()  

    
def detect_image(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()
    k=[]

    #print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:

        draw(image, boxes, scores, classes, all_classes)
        print(classes)
        #classes = classes[0].strip(' ')
        print(len(classes))
        for i in range(len(classes)):
            print(all_classes[classes[i]])
            if all_classes[classes[i]] not in k:
                k.append(all_classes[classes[i]])
    return image


def detect_video( yolo, all_classes):
    """Use yolo v3 to detect video.
    # Argument:
        video: video file.
        yolo: YOLO, yolo model.
        all_classes: all classes name.
    """
    # ipv4 address
    ipv4_url = 'http://192.168.171.44:8080'
    # read video
    cam = f'{ipv4_url}/video'
    camera= cv2.VideoCapture(cam)
    while True:
        res, frame = camera.read()
        frame = cv2.resize(frame, (640, 480))
        if not res:
            break
        image = detect_image(frame, yolo, all_classes)
        cv2.imshow("Video Stream", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    yolo = YOLO(0.6, 0.5)
    file = 'data/coco_classes.txt'
    all_classes = get_classes(file)
    detect_video( yolo, all_classes)