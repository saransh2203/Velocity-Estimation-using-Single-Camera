import cv2
import time
from ultralytics import YOLO

# variables
initialTime = {}
initialDistance = {}
changeInTime = 0
changeInDistance = 0

listDistance = []
listspeed = []
# distance from camera to object(face) measured
KNOWN_DISTANCE = 500  # centimeter
# width of face in the real world or Object Plane
KNOWN_WIDTH = 250  # centimeter
WIDTH_DICT = {
    'HMV': 175,
    'LMV': 175,
    '2-Wheeler': 14.3,
    'Pedestrian': 14.3,
    'Auto': 14.3,
    'Animal': 14.3,
    'car': 250,
    'truck': 350,
    'suitcase': 50,
    'airplane': 1000,
    'person': 10,
    'bus': 350,
    'train': 400,
    'motorcycle': 20,
    'stop sign': 10,
    'kite': 5,
    'tie':1,
}
DANGER_DIST = 6000  # centimeter
# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
fonts = cv2.FONT_HERSHEY_COMPLEX
cap = cv2.VideoCapture('pexels_videos_4516 (1080p).mp4')
Distance = 0
# face detector object
face_detector = YOLO('yolov8s.pt')


# focal length finder function
def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value


# distance estimation function
def distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance


# face detector function
def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), WHITE, 1)
        face_width = w

    return face_width


def yolo_to_opencv(yolo_boxes):
    x, y, w, h = yolo_boxes
    left = int((x - w / 2))
    right = int(w)
    top = int((y - h / 2))
    bottom = int(h)
    return left, top, right, bottom


def face_data1(image):
    face_widths = []  # List to store face widths
    face_index = 1  # Counter for assigning an integer index to each face
    infoz = []

    prid = face_detector.predict(image)
    for r in prid:
        boxes = r.boxes
        for box in boxes:
            b = box.xywh[0]  # get box coordinates in (top, left, bottom, right) format
            (x, y, h, w) = yolo_to_opencv(b)
            face_width = w
            face_widths.append(face_width)

            c = face_detector.names[int(box.cls)]
            infoz.append([x, y, h, w, c])
            face_index += 1

    return face_widths, infoz


def display_info_on_box(image, index, face_coordinates, speed, distance, color):
    (x, y, w, h, c) = face_coordinates[index]

    # Draw the box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Display speed and distance on top of the box
    text = f"{index} {c} Speed: {round(speed, 2)}m/s | Distance: {distance:.2f}cm"
    text_size, _ = cv2.getTextSize(text, fonts, 0.9, 1)
    text_x = x + (w // 2) - (text_size[0] // 2)
    text_y = y - 10
    cv2.putText(image, text, (text_x, text_y), fonts, 0.9, color, 2, cv2.LINE_AA)


def speedFinder(coveredDistance, timeTaken):
    speed = coveredDistance / timeTaken
    return speed


def averageFinder(completeList, averageOfItems):
    lengthOfList = len(completeList)
    selectedItems = lengthOfList - averageOfItems
    selectedItemsList = completeList[selectedItems:]
    average = sum(selectedItemsList) / len(selectedItemsList)
    return average


# reading reference image from directory
ref_image = cv2.imread("Ref_image.png")

ref_image_face_width, loc = face_data1(ref_image)
focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width[0])
print(focal_length_found)
cv2.imshow("ref_image", ref_image)

# Get the width and height of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video filename and codec (for example, MP4V).
output_filename = 'video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Define the VideoWriter object
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

while True:
    _, frame = cap.read()

    # calling face_data function
    face_width_in_frame, loc = face_data1(frame)
    # finding the distance by calling function Distance
    err = len(face_width_in_frame)
    if err != 0:
        listDistance = [[] for _ in range(err)]
        distanceInMeters = [None] * err
        listspeed = [[] for _ in range(err)]

        for i in range(err):
            riyal_width = WIDTH_DICT[loc[i][4]]
            Distance = distance_finder(focal_length_found, riyal_width, face_width_in_frame[i])
            listDistance[i].append(Distance)
            averageDistance = averageFinder(listDistance[i], 12)
            # cm to m
            distanceInMeters[i] = averageDistance / 100

            if i in initialDistance:  # Check if the face is already in the dictionary
                # finding the change in distance
                changeInDistance = initialDistance[i] - distanceInMeters[i]

                if changeInDistance < 0:
                    changeInDistance *= -1

                # finding change in time
                changeInTime = time.time() - initialTime[i]

                # finding the speed
                speed = speedFinder(coveredDistance=changeInDistance, timeTaken=changeInTime)
                listspeed[i].append(speed)
                averageSpeed = averageFinder(listspeed[i], 40)

                if Distance <= DANGER_DIST:
                    display_info_on_box(frame, i, loc, round(averageSpeed, 2), round(averageDistance, 2), GREEN)
                else:
                    display_info_on_box(frame, i, loc, round(averageSpeed, 2), round(Distance, 2), GREEN)

            # update initial distance and time in the dictionary
            initialDistance[i] = distanceInMeters[i]
            initialTime[i] = time.time()

    # Write the processed frame to the output video
    out.write(frame)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Release the VideoWriter object, release the input video, and close all windows
out.release()
cap.release()
cv2.destroyAllWindows()
