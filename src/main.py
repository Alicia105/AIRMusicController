import cv2
import threading
import mediapipe as mp
import numpy as np
import detection
import audio_processing
import shared_data
import time
from audio_processing import start_audio_system, control_audio_vision

#import hands landmarks and medeiapipe hand tracking model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

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

def draw_volume(frame,hand,action):
    #global volume
    with shared_data.param_lock:
        vol=shared_data.volume
    # Define bar dimensions
    bar_x = 450         # x position of the bar
    bar_y = 50          # y position (top of the bar)
    bar_width = 30      # width of the bar
    bar_height = 380    # max height of the bar
    if action=="Volume":
        vol=set_graduation(frame,hand,bar_y, bar_y + bar_height,0,2.0,10,2)
    volume_level = (vol/2.0) 

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
    return volume_level
    
def draw_pitch(frame,hand,action):
    #global pitch_shift_steps
    with shared_data.param_lock:
        pitch_shift=shared_data.pitch_shift_steps

    # Define bar dimensions
    bar_x = 530          # x position of the bar
    bar_y = 50           # y position (top of the bar)
    bar_width = 30       # width of the bar
    bar_height = 380     # total height of the bar

    # Get semitone shift from hand
    if action=="Pitch":
        pitch_shift = set_graduation(frame, hand, bar_y, bar_y + bar_height, -12, 12, 24, 2)
    
    # Clamp pitch shift just in case
    pitch_shift = max(-12, min(12, pitch_shift))
    
    # Calculate pitch multiplier
    pitch_multiplier = round(2 ** (pitch_shift / 12), 3)

    # Normalize filled bar height
    normalized = (pitch_shift + 12) / 24  # maps -12:12 --> 0:1
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
    cv2.putText(frame, f'{pitch_shift:+} st', (bar_x - 15, bar_y + bar_height + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'x{pitch_multiplier}', (bar_x - 15, bar_y + bar_height + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return pitch_shift

def draw_speed(frame,hand,action):
    #global speed_rate
    with shared_data.param_lock:
        speed=shared_data.speed_rate
    # Define bar dimensions
    bar_x = 600          # x position of the bar
    bar_y = 50          # y position (top of the bar)
    bar_width = 30      # width of the bar
    bar_height = 380    # max height of the bar
    if action=="Speed":
        speed=set_graduation(frame,hand,bar_y, bar_y + bar_height,0.5,2.0,10,2)
    speed_level = (speed-0.5)/(2.0-0.5)

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
    cv2.putText(frame, f'{int((speed_level)* 100)}%', (bar_x - 10, bar_y + bar_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)    
    return speed

def handle_dash_board(frame,hand,action):
    # === DRAWING DASHBOARD ===
    vol=draw_volume(frame,hand,action)
    pitch=draw_pitch(frame,hand,action)
    speed=draw_speed(frame,hand,action)
    return vol,pitch,speed

def main():
    last_update = time.time()
    update_interval = 0.2  # seconds

    # Start audio system in vision-only control mode
    threading.Thread(target=audio_processing.start_audio_system, daemon=True).start()
    
    # Start audio output
    cap=cv2.VideoCapture(0)
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

            #Rendering results 
            # Color in BGR in DrawingSpec 
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(255,255,120), thickness=2, circle_radius=4),
                                                    mp_drawing.DrawingSpec(color=(255,76,134), thickness=2, circle_radius=2))
                    
                    new_volume=shared_data.volume
                    new_pitch=shared_data.pitch_shift_steps
                    new_speed=shared_data.speed_rate
                    new_playing=shared_data.is_playing
                        
                        #Render Left or right hand label
                    if get_hand_label(num, hand, results,width,height) and len(results.multi_hand_landmarks)==2:
                        text, coord = get_hand_label(num, hand, results,width,height)
                        cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
                        name_hand=text.split()[0]  # Gets "Left" or "Right"
                        print(name_hand)
                        now=time.time()

                        #use left hand for audio player
                        if name_hand=="Left":
                            t=detection.control_audio_player(hand)
                            if t=="Pause":
                                with shared_data.param_lock:
                                    shared_data.is_playing = False
                                new_playing=False
                            if t=="Play":
                                with shared_data.param_lock:
                                    shared_data.is_playing = True
                                new_playing=True
                            print_message(image,t,1)

                        #use right hand for audio controller
                        if name_hand=="Right":
                            draw_controller(image,hand,landmark_id)
                            action=detection.get_actions(hand)
                            print_message(image,action,2)
                            new_volume,new_pitch,new_speed=handle_dash_board(image,hand,action)

                    #if unique hand use it for controller        
                    elif len(results.multi_hand_landmarks)==1:
                        now=time.time()
                        draw_controller(image,hand,landmark_id)
                        action=detection.get_actions(hand)
                        print_message(image,action,2)
                        new_volume,new_pitch,new_speed=handle_dash_board(image,hand,action)
                        

                    #if too much hands
                    elif len(results.multi_hand_landmarks)>2:
                        txt="Too much hands on screen"
                        cv2.putText(image, txt,(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                    
                    if now - last_update > update_interval:
                        control_audio_vision(new_volume,new_pitch,new_speed,new_playing)
                        last_update = time.time()
                print(f"Volume={shared_data.volume:.2f}, Pitch steps={shared_data.pitch_shift_steps}, Speed={shared_data.speed_rate:.2f}, isPlaying={shared_data.is_playing}")
            
            
            cv2.imshow("AIR Music Controller",image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                with shared_data.param_lock:
                    shared_data.is_playing = not shared_data.is_playing
                control_audio_vision(shared_data.volume,shared_data.pitch_shift_steps,shared_data.speed_rate,shared_data.is_playing)

        cap.release()
        cv2.destroyAllWindows()
        print(f"Frame size: {width} x {height}")

        with shared_data.param_lock:
            shared_data.is_playing = False  # Graceful shutdown
        control_audio_vision(shared_data.volume,shared_data.pitch_shift_steps,shared_data.speed_rate,shared_data.is_playing)

    
    # Keep main alive
    """except KeyboardInterrupt:
        audio_processing.stop_stream()
        print("Stopped by user.")"""

if __name__ == "__main__":
    main()





