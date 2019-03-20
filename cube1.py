import cv2
import numpy as np
import math

def pixelDistance(A, B):
    """
    Use Distance formula D=sqrt((y2-y1)^2 +(x2-x1)^2)
    """
    (col_A, row_A) = A
    (col_B, row_B) = B

    return math.sqrt(math.pow(col_B - col_A, 2) + math.pow(row_B - row_A, 2))


def sortCorners(cor1, cor2, cor3, cor4):
    """
    The corners are labeled
        A ---- B
        |      |
        |      |
        C ---- D
        
    Sort the corners and make them as per label
    - A is top left
    - B is top right
    - C is bottom left
    - D is bottom right
    
    Return result containing (A, B, C, D) tuple
    """
    result = []
    corners = (cor1, cor2, cor3, cor4)

    min_x=None
    max_x=None
    min_y=None
    max_y=None           
    

    for (x,y) in corners:
        if min_x == None or min_x > x:
            min_x = x

        if max_x == None or max_x < x:
            max_x = x

        if min_y == None or min_y > y:
            min_y = y

        if max_y == None or max_y < y:
            max_y = y

    
    # A is top left
    top_left = None
    top_left_distance = None
    for (x, y) in corners:
        distance = pixelDistance((min_x, min_y), (x, y))
        if top_left_distance is None or distance < top_left_distance:
            top_left = (x, y)
            top_left_distance = distance

    result.append(top_left)

    # B is top right
    top_right=None
    top_right_distance=None
    for (x, y) in corners:
        if (x, y) in result:
            continue

        distance=pixelDistance ((max_x,min_y), (x,y))
        if top_right_distance == None or distance < top_right_distance:
            top_right = (x, y)
            top_right_distance = distance

    result.append(top_right)

    # C is bottom left
    bottom_left=None
    bottom_left_distance=None
    for (x, y) in corners:
        if (x, y) in result:
            continue

        distance=pixelDistance ((min_x,max_y), (x,y))
        if bottom_left_distance == None or distance < bottom_left_distance:
            bottom_left = (x, y)
            bottom_left_distance = distance

    result.append(bottom_left)

    # D is bottom right
    bottom_right=None
    bottom_right_distance=None
    for (x, y) in corners:
        if (x, y) in result:
            continue

        distance=pixelDistance ((max_x,max_y), (x,y))
        if bottom_right_distance == None or distance < bottom_right_distance:
            bottom_right = (x, y)
            bottom_right_distance = distance

    result.append(bottom_right)


    return result


def findAngle(A, B, C):
    """
    Return the angle at A (in radians) for the triangle formed by A, B, C
    a, b, c are lengths
    Formula cos (A) = b^2 + c^2 - a^2
                    -------------------
                        2 * b * c

        A
       / \
    c /   \b
     /     \
    B-------C
        a
    """

    (col_A, row_A) = A
    (col_B, row_B) = B
    (col_C, row_C) = C

    length_CB = pixelDistance(C ,B)
    length_AB = pixelDistance(A ,B)
    length_CA = pixelDistance(C ,A)

    cos_A = (math.pow(length_CA, 2) + math.pow(length_AB, 2) - math.pow(length_CB , 2)) / (2 * length_CA * length_AB)
    angle_BAC = math.acos(cos_A)
    return math.degrees(angle_BAC)
    
def approxIsSquare(approx, length_threshold = 0.7, angle_threshold = 10):
    """
    Task:
    1. All lines of same length.
    2. All four corners must be of 90 degree approx.

    SIDE_VS_SIDE_THRESHOLD
        If this is 1 then all 4 sides must be the exact same length.  If it is
        less than one that all sides must be within the percentage length of
        the longest side.

    ANGLE_THRESHOLD
        If this is 0 then all 4 corners must be exactly 90 degrees.  If it
        is 10 then all four corners must be between 80 and 100 degrees.

    The corners are labeled
        A ---- B
        |      |
        |      |
        C ---- D
    """

    #Sort four corners according to label
    (A,B,C,D) = sortCorners(tuple(approx[0][0]),
                             tuple(approx[1][0]),
                             tuple(approx[2][0]),
                             tuple(approx[3][0]))

    # To find lenght of each sides
    length_AB = pixelDistance(A, B)
    length_AC = pixelDistance(A, C)
    length_CD = pixelDistance(C, D)
    length_DB = pixelDistance(D, B)

    max_length = max(length_AB, length_AC, length_CD, length_DB)
    cutoff_length = max_length * length_threshold

    for length in (length_AB, length_AC, length_CD, length_DB):
        if length < cutoff_length:
            return False

    # min and max angle for cutoff
    min_angle = 90 - angle_threshold
    max_angle = 90 + angle_threshold

    # To find angle of each corners
    # Angle at A
    angle_A = int(findAngle(A, B, C))
    if angle_A < min_angle or angle_A > max_angle:
        return False
    
    # Angle at B
    angle_B = int(findAngle(B, A, D))
    if angle_B < min_angle or angle_B > max_angle:
        return False

    # Angle at C
    angle_C = int(findAngle(C, D, A))
    if angle_C < min_angle or angle_C > max_angle:
        return False

    # Angle at D
    angle_D = int(findAngle(D, B, C))
    if angle_D < min_angle or angle_D > max_angle:
        return False

    
    return True

def squareApproximation(contours):
    square_contours = []
    """
    Contours those are square approx
    """
    for cnt in contours:
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        if len(approx) == 4:
            if approxIsSquare(approx):
                square_contours.append(approx)
    return square_contours

def searchSquareInSquare(square_contours):
    square_in_square = []
    """
    Finding smaller squares those are in big squares
    """
    for j in range(0,len(square_contours)):
        check=0
        (x, y, w, h) = cv2.boundingRect(square_contours[j])
        for i in range(j+1,len(square_contours)):
            (xx, yy, ww, hh) = cv2.boundingRect(square_contours[i])
            if (x<=xx) and ((x+w)>=(xx+ww)) and (y<yy) and ((y+h)>=(yy+hh)) \
               and cv2.contourArea(square_contours[j])/25 > cv2.contourArea(square_contours[i]):
                square_in_square.append(square_contours[i])
            elif (x<=xx) and ((x+w)>=(xx+ww)) and (y<yy) and ((y+h)>=(yy+hh)):
                square_in_square.append(square_contours[j])

    """
    Removing duplicacy from square_in_square
    """

    j = 0
    while j < len(square_in_square):
        i = j + 1
        while i < len(square_in_square):
            if (square_in_square[j]==square_in_square[i]).all():
                del square_in_square[i]
                i = i - 1
            i = i + 1
        j = j + 1

    return square_in_square

def eliminatingSquares(square_contours, square_in_square):
    """
    Removing smaller squares from the list and keeping rest
    """

    temp_sc_copy = square_contours.copy()


    if(len(square_in_square)>0):
        j = 0
        while j < len(temp_sc_copy):
            i = 0
            while i < len(square_in_square):
                if (temp_sc_copy[j]==square_in_square[i]).all():
                    del temp_sc_copy[j]
                    j = j - 1
                    break
                i = i + 1
            j = j + 1

    sc_sis = temp_sc_copy

        
    ##print (square_contours)
    ##print (square_in_square)
    ##print (sc_sis)
    ##print (temp_sc_copy)



    """ 
    For finding which are cells of rubik's cube face, I am going to
    find area and keep only those which are of same area in approx and
    remove significantly small ones. (For starter)

    """
    max_area=0

    for sc in sc_sis:
        area = cv2.contourArea(sc)
        if max_area < area:
            max_area = area

    cube_cells = []
    for sc in sc_sis:    
        area = cv2.contourArea(sc)
        if max_area/4 < area:
            cube_cells.append(sc)
    return cube_cells
    

image=cv2.imread("cube.png")

"""
Canny Edge Detection
"""
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
canny = cv2.Canny(blurred, 20, 40)
##cv2.imshow("try",canny)

"""
Dilation
"""
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(canny, kernel, iterations=3)
##cv2.imshow("3",dilated)

"""
Finding Contours
"""
(contours, hierarchy) = cv2.findContours(dilated.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##cv2.drawContours(image, contours, -1, (255,0,0), 2)
##cv2.imshow("1",image)

square_contours = squareApproximation (contours)
square_in_square = searchSquareInSquare(square_contours)  
cube_cells = eliminatingSquares(square_contours, square_in_square)



cccc=0
for sc in cube_cells:
    cv2.drawContours(image, [sc], -1, (0,255,0), 2)
    cv2.putText(image,str(cccc),cv2.boundingRect(sc)[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)
    cccc=cccc+1

cv2.imshow("5",image)


cv2.waitKey(0)
cv2.destroyAllWindows()

