import mediapipe as mp
import cv2
import numpy as np
import pyautogui
import detection

#import hands landmarks and medeiapipe hand tracking model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap=cv2.VideoCapture(0)

"""index: the hand result (i.e 0 or 1), hand: the actual hand landmarks, results: all detections from model"""
def get_hand_label(index,hand,results,width,height):
    output=None
    for idx,classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index :

            #Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label,round(score,3))

            #Extract coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x,hand.landmark[mp_hands.HandLandmark.WRIST].y)),[width,height]).astype(int))
            output = text, coords
    return output

def draw_controller(img,hand,landmark_id):
    lm = hand.landmark[landmark_id]

    h, w, c = img.shape
    cx, cy = int(lm.x * w), int(lm.y * h)
    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

def get_controller_screen_coordinates(hand,landmark_id):
    # Get screen size
    screen_w, screen_h = pyautogui.size()

    lm = hand.landmark[landmark_id]
    # Convert normalized coordinates to screen space
    x_screen = int(lm.x * screen_w)
    y_screen = int(lm.y * screen_h)

    return [x_screen,y_screen]

def print_message(image,selector):
    match selector:
        case 0:
            text="Mouse moving"
            color=(0,255,0)
        case 1:
            text="Left click"
            color=(255,0,0)          
        case 2:
            text="Right click"
            color=(80,102,227)                
        case 3:
            text="Double click"
            color=(0,0,255)
        case _:
            return None   

    cv2.putText(image, text,(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
    return 
           

with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands :
    while cap.isOpened():
        # Get index tip (id 8)
        landmark_id = 8 

        ret,frame=cap.read()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        

        #Flip horizontally
        frame=cv2.flip(frame,1)

        #convert BGR to RGB-->necessary to use mediapipe 
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        #set flags
        frame_rgb.flags.writeable=False

        #Detections
        results=hands.process(frame_rgb)

        #Set flag to true
        frame_rgb.flags.writeable=True

        #Convert RGB back to BGR
        image=cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR)

        print(results)
        #Rendering results 
        # Color in BGR in DrawingSpec 
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255,255,120), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(255,76,134), thickness=2, circle_radius=2))
                
                #Render Left or right hand label
                if get_hand_label(num, hand, results,width,height):
                    text, coord = get_hand_label(num, hand, results,width,height)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
                    screen_coordinates = get_controller_screen_coordinates(hand,landmark_id)
                    x_screen = screen_coordinates[0]
                    y_screen = screen_coordinates[1]

                    text = text.split()
                    name_hand=text[0]
                    print(name_hand)

                    #use right hand for cursor
                    if name_hand=="Right":
                        draw_controller(image,hand,landmark_id)
                        selector=detection.interactWithScreen(hand,x_screen, y_screen)
                        print_message(image,selector)
                #use unique hand for cursor        
                if len(results.multi_hand_landmarks)==1:
                    screen_coordinates = get_controller_screen_coordinates(hand,landmark_id)
                    x_screen = screen_coordinates[0]
                    y_screen = screen_coordinates[1]
                    draw_controller(image,hand,landmark_id)
                    selector=detection.interactWithScreen(hand,x_screen, y_screen)
                    print_message(image,selector)
                if len(results.multi_hand_landmarks)>2:
                    txt="Too much hands on screen"
                    cv2.putText(image, txt,(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                   
        cv2.imshow("Hand Tracking",image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()

cv2.destroyAllWindows()

print(f"Frame size: {width} x {height}")






