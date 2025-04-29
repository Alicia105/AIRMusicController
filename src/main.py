import mediapipe as mp
import cv2
import numpy as np
import pyautogui
import detection

#import hands landmarks and medeiapipe hand tracking model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

volume = 1.0
pitch_shift_steps = 0
speed_rate = 1.0

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

def print_message(image,text,selector):
    if selector==1:
        color=(0,255,0)
        cv2.putText(image, text,(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)

    if selector==2:
        color=(0,0,255)
        cv2.putText(image, text,(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
    return 

def set_graduation(image, hand, minPix, maxPix, minVal, maxVal, numGrad, selector):
    lm = hand.landmark[8]  # Index finger tip

    h, w, c = image.shape
    x, y = int(lm.x * w), int(lm.y * h)

    incrPixel = (maxPix - minPix) / numGrad
    incrValues = (maxVal - minVal) / numGrad

    if selector == 2:
        for i in range(numGrad):
            low = minPix + i * incrPixel
            high = minPix + (i + 1) * incrPixel
            if low <= y < high:
                return minVal + i * incrValues

    if selector == 1:
        for i in range(numGrad):
            low = minPix + i * incrPixel
            high = minPix + (i + 1) * incrPixel
            if low <= x < high:
                return minVal + i * incrValues

    return minVal  # Default if no match

def draw_volume(frame,hand):
    global volume
    
    # Define bar dimensions
    bar_x = 580          # x position of the bar
    bar_y = 60          # y position (top of the bar)
    bar_width = 30      # width of the bar
    bar_height = 380    # max height of the bar

    volume=set_graduation(frame,hand,bar_y, bar_y + bar_height,0,2.0,10,2)
    volume_level = (volume/2.0) 

    # Calculate the current filled height based on volume
    filled_height = int(bar_height * volume_level)

    cv2.putText(frame,'Volume', (bar_x - 30, bar_y -10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw the background of the bar (empty part)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), thickness=-1)

    # Draw the filled part (current volume)
    cv2.rectangle(frame, (bar_x, bar_y + (bar_height - filled_height)), 
                    (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), thickness=-1)

    # Add a volume percentage text
    cv2.putText(frame, f'{int(volume_level * 100)}%', (bar_x - 10, bar_y + bar_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)    
    


def draw_pitch(frame, hand):
    global pitch_shift_steps

    # Define bar dimensions
    bar_x = 570          # x position of the bar
    bar_y = 60           # y position (top of the bar)
    bar_width = 30       # width of the bar
    bar_height = 380     # total height of the bar

    # Get semitone shift from hand
    pitch_shift_steps = set_graduation(frame, hand, bar_y, bar_y + bar_height, -12, 12, 24, 2)
    
    # Clamp pitch shift just in case
    pitch_shift_steps = max(-12, min(12, pitch_shift_steps))
    
    # Calculate pitch multiplier
    pitch_multiplier = round(2 ** (pitch_shift_steps / 12), 3)

    # Normalize filled bar height
    normalized = (pitch_shift_steps + 12) / 24  # maps -12:12 --> 0:1
    filled_height = int(bar_height * normalized)

    # Draw the background of the bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), thickness=-1)

    # Draw the filled part
    cv2.rectangle(frame, (bar_x, bar_y + (bar_height - filled_height)),
                  (bar_x + bar_width, bar_y + bar_height), (255, 0, 0), thickness=-1)

    # Draw graduation marks
    num_grads = 24  # semitones from -12 to +12
    grad_spacing = bar_height / num_grads
    for i in range(num_grads + 1):
        y = int(bar_y + i * grad_spacing)
        if (i - 12) % 12 == 0:  
            # Thicker graduation for 0 (no shift) and +-12
            line_length = 20
            color = (255, 255, 255)
            thickness = 2
        else:
            line_length = 10
            color = (200, 200, 200)
            thickness = 1
        
        cv2.line(frame, (bar_x + bar_width + 2, y), (bar_x + bar_width + 2 + line_length, y), color, thickness)

    # Title
    cv2.putText(frame, 'Pitch', (bar_x - 20, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display pitch shift value
    cv2.putText(frame, f'{pitch_shift_steps:+} st', (bar_x - 15, bar_y + bar_height + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'x{pitch_multiplier}', (bar_x - 15, bar_y + bar_height + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_speed(frame,hand):

    # Define bar dimensions
    bar_x = 580          # x position of the bar
    bar_y = 60          # y position (top of the bar)
    bar_width = 30      # width of the bar
    bar_height = 380    # max height of the bar

    speed_rate=set_graduation(frame,hand,bar_y, bar_y + bar_height,0.5,2.0,10,2)
    speed_level = (speed_rate/(2.0-0.5)) 

    # Calculate the current filled height based on volume
    filled_height = int(bar_height * speed_level)

    cv2.putText(frame,'Speed', (bar_x - 25, bar_y -10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw the background of the bar (empty part)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), thickness=-1)

    # Draw the filled part (current volume)
    cv2.rectangle(frame, (bar_x, bar_y + (bar_height - filled_height)), 
                    (bar_x + bar_width, bar_y + bar_height), (0, 0, 255), thickness=-1)

    # Add a volume percentage text
    cv2.putText(frame, f'{int(speed_level * 100)}%', (bar_x - 10, bar_y + bar_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)    

def handle_dash_board(frame,hand,action):
    # === DRAWING DASHBOARD ===
    lm = hand.landmark[8]

    h, w, c = frame.shape
    x, y = int(lm.x * w), int(lm.y * h)

    if action=="Volume":
        draw_volume(image,hand)
    if action=="Pitch":
        draw_pitch(image,hand)

    if action=="Speed":
        draw_speed(image,hand)

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
                    text = text.split()
                    name_hand=text[0]
                    print(name_hand)

                    #use left hand for audio player
                    if name_hand=="Left":
                        t=detection.control_audio_player(hand)
                        print_message(image,t,1)

                    #use right hand for audio controller
                    if name_hand=="Right":
                        draw_controller(image,hand,landmark_id)
                        action=detection.get_actions(hand)
                        print_message(image,action,2)
                        handle_dash_board(image,hand,action)
                        
                #if unique hand use it for controller        
                if len(results.multi_hand_landmarks)==1:
                    draw_controller(image,hand,landmark_id)
                    action=detection.get_actions(hand)
                    print_message(image,action,2)
                    handle_dash_board(image,hand,action)

                #if too much hands
                if len(results.multi_hand_landmarks)>2:
                    txt="Too much hands on screen"
                    cv2.putText(image, txt,(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        
                   
        cv2.imshow("AIR Music Controller",image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()

cv2.destroyAllWindows()

print(f"Frame size: {width} x {height}")






