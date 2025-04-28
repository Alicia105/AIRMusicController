import numpy as np

#with a,b,c being points
#Calculate the euclidian norm between two points 
def finger_joints_coordinates(hand,idx1,idx2,idx3,idx4):
    x1, y1, z1 = hand.landmark[idx1].x, hand.landmark[idx1].y, hand.landmark[idx1].z
    x2, y2, z2 = hand.landmark[idx2].x, hand.landmark[idx2].y, hand.landmark[idx2].z
    x3, y3, z3 = hand.landmark[idx3].x, hand.landmark[idx3].y, hand.landmark[idx3].z
    x4, y4, z4 = hand.landmark[idx4].x, hand.landmark[idx4].y, hand.landmark[idx4].z
    a = [x1, y1, z1]
    b = [x2, y2, z2]
    c = [x3, y3, z3]
    d = [x4, y4, z4]

    return [a,b,c,d]

def get_finger(hand,name=""):
    if(name=="thumb"):
        return finger_joints_coordinates(hand,4,3,2,1)
    elif(name=="index"):
        return finger_joints_coordinates(hand,8,7,6,5)
    elif(name=="middle"):
        return finger_joints_coordinates(hand,12,11,10,9)
    elif(name=="ring"):
        return finger_joints_coordinates(hand,16,15,14,13)
    elif(name=="pinky"):
        return finger_joints_coordinates(hand,17,18,19,20)
    return None

def get_distance(a,b):
    norm=np.hypot(b[0]-a[0],b[1]-a[1])
    result=np.interp(norm,[0,1],[0,1000])
    return result

def get_angle(a,b,c):
    gama=np.arctan2(c[1]-b[1],c[0]-b[0])
    beta=np.arctan2(a[1]-b[1],a[0]-b[0])
    alpha=beta-gama
    r=np.degrees(alpha)
    return abs(r)

def isPalmFingerBent(hand,name):
    finger=get_finger(hand,name)
    if finger!=None:
        a=finger[0]
        b=finger[2]
        c=finger[3]
        alpha=get_angle(a,b,c)
        if alpha<90:
            return True
        return False
    return None #fallback

def isThumbBent(hand):
    thumb=get_finger(hand,name="thumb")
    index=get_finger(hand,name="index")
    if thumb!=None and index!=None:
        a=thumb[0]
        b=thumb[2]
        c=index[3]
        alpha=get_angle(a,b,c)
        if alpha<20 and get_distance(a,c)<50:
            return True
        return False
    return None #fallback

#Hand gestures for audio player

#Pause
def isFist(hand):
    if(isThumbBent(hand) and isPalmFingerBent(hand,name="index") and isPalmFingerBent(hand,name="middle") and isPalmFingerBent(hand,name="ring") and isPalmFingerBent(hand,name="pinky")):
        return True
    return False

#restart track 
def isPeace(hand):
    if(isThumbBent(hand) and not isPalmFingerBent(hand,name="index") and not isPalmFingerBent(hand,name="middle") and isPalmFingerBent(hand,name="ring") and isPalmFingerBent(hand,name="pinky")):
        return True
    return False

#skip track 
def isThumbUp(hand):
    if(not isThumbBent(hand) and isPalmFingerBent(hand,name="index") and isPalmFingerBent(hand,name="middle") and isPalmFingerBent(hand,name="ring") and isPalmFingerBent(hand,name="pinky")):
        return True
    return False

#rewind track 
def isPinkyUp(hand):
    if(not isThumbBent(hand) and isPalmFingerBent(hand,name="index") and isPalmFingerBent(hand,name="middle") and isPalmFingerBent(hand,name="ring") and not isPalmFingerBent(hand,name="pinky")):
        return True
    return False

#Play
def isOpenHand(hand):
    if(not isThumbBent(hand) and not isPalmFingerBent(hand,name="index") and not isPalmFingerBent(hand,name="middle") and not isPalmFingerBent(hand,name="ring") and not isPalmFingerBent(hand,name="pinky")):
        return True
    return False

#open equalizer menu ?
def isLoveU(hand):
    if(not isThumbBent(hand) and not isPalmFingerBent(hand,name="index") and isPalmFingerBent(hand,name="middle") and isPalmFingerBent(hand,name="ring") and not isPalmFingerBent(hand,name="pinky")):
        return True
    return False

#Hand gestures for audio controller


def get_gesture_name(hand):
    t="None"
    if isFist(hand):
        t="Fist"
    if isPeace(hand):
        t="Peace"
    if isLoveU(hand):
        t="I Love U"
    if isOpenHand(hand):
        t="Open Hand"
    return t

"""
def control_audio_parameters(hand):
    if isFist(hand):
    if isPeace(hand):
    if isThumbUp(hand):
    if isPinkyUp(hand):
    if isOpenHand(hand):
    if isLoveU(hand):"""

def control_audio_player(hand):
    t="None"
    if isFist(hand):
        t="Pause"
    if isPeace(hand):
        t="Restart"
    if isThumbUp(hand):
        t="Skip"
    if isPinkyUp(hand):
        t="Rewind"
    if isOpenHand(hand):
        t="Play"
    if isLoveU(hand):
        t="Equalizer"
    return t


def handleVolume(hand):
    if(isThumbBent(hand) and not isPalmFingerBent(hand,name="index") and isPalmFingerBent(hand,name="middle") and isPalmFingerBent(hand,name="ring") and isPalmFingerBent(hand,name="pinky")):
        return True
    return False

def handlePitch(hand):
    if(isThumbBent(hand) and not isPalmFingerBent(hand,name="index") and not isPalmFingerBent(hand,name="middle") and isPalmFingerBent(hand,name="ring") and isPalmFingerBent(hand,name="pinky")):
        return True
    return False

def handleSpeed(hand):
    if(isThumbBent(hand) and not isPalmFingerBent(hand,name="index") and not isPalmFingerBent(hand,name="middle") and not isPalmFingerBent(hand,name="ring") and isPalmFingerBent(hand,name="pinky")):
        return True
    return False


def get_actions(hand):
    t="None"
    if handleVolume(hand):
        t="Volume"
    if handleSpeed(hand):
        t="Speed"
    if handlePitch(hand):
        t="Pitch"
    return t