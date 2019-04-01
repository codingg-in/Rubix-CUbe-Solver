import cv2
import numpy as np
import math
import imutils
import os

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

def minX(cube_cells, j):
    min_x = None
    pos = 0
    for i in range (j, len(cube_cells)):
        x = cube_cells[i][0][0][0]
        y = cube_cells[i][0][0][1]
        if min_x == None or x < min_x:
            min_x = x
            pos = i

    return min_x, pos
    
def minY(cube_cells, j):
    min_y = None
    pos = 0
    for i in range (j, len(cube_cells)):
        x = cube_cells[i][0][0][0]
        y = cube_cells[i][0][0][1]
        if min_y == None or y < min_y:
            min_y = y
            pos = i

    return min_y, pos

def maxX(cube_cells, j):
    max_x = None
    pos = 0
    for i in range (j, len(cube_cells)):
        x = cube_cells[i][0][0][0]
        y = cube_cells[i][0][0][1]
        if max_x == None or x > max_x:
            max_x = x
            pos = i

    return max_x, pos

def maxY(cube_cells, j):
    max_y = None
    pos = 0
    for i in range (j, len(cube_cells)):
        x = cube_cells[i][0][0][0]
        y = cube_cells[i][0][0][1]
        if max_y == None or y > max_y:
            max_y = y
            pos = i

    return max_y, pos

def calculateThresholdForOrientation(cube_cells):
    """
    This is important function to calculate threshold value for orientation
    of cells as if image of cube is in some rotated orientation then this will
    adjust its algorithm accordingly.
    """
    arr_x = []
    arr_y = []
    thres = 0
    for i in range (0, len(cube_cells)):
        x = cube_cells[i][0][0][0]
        y = cube_cells[i][0][0][1]
        arr_x.append(x) 
        arr_y.append(y)
    for j in range (0, len(arr_y)):
        ty = None
        pos = 0
        for i in range (j, len(arr_y)):
            if ty == None or arr_y[i] < ty:
                ty = arr_y[i]
                pos = i
        temp = arr_x[j]
        del arr_x[j]
        arr_x.insert(j,arr_x[pos-1])
        del arr_x[pos]
        arr_x.insert(pos,temp)
        temp = arr_y[j]
        del arr_y[j]
        arr_y.insert(j,arr_y[pos-1])
        del arr_y[pos]
        arr_y.insert(pos,temp)

    i = 0
    sqrt_cc = int(math.sqrt(len(cube_cells)))
    floor_cc = math.sqrt(len(cube_cells))

    if floor_cc - math.floor(floor_cc) < 1.0 and floor_cc - math.floor(floor_cc) > 0:
        perfect_sq = sqrt_cc + 1
    else:
        perfect_sq = sqrt_cc
        
    while i < len(cube_cells) - sqrt_cc + 1 :
        count = 0
        if  i + perfect_sq > len(cube_cells) and sqrt_cc != perfect_sq:
            break
        for j in range (1, perfect_sq):
            thres = max (thres, abs(arr_y[i + j] - arr_y[i + j - 1]))
        thres = max (thres, abs(arr_y[i + j] - arr_y[i + 0]))
        i = i + perfect_sq

    return thres + 1

def orientation(cube_cells):
    """
    Arranging cell at (1, 1) coordinate at the first position of cube_cells,
    cell at (1, 2) coordinate at the second position of cube_cells so on...
    """

##    for k in range (0, len(cube_cells)):
##        (x, y, w, h) = cv2.boundingRect(cube_cells[k])
##        print(x,".",y)
##    print ("\n")
    threshold = calculateThresholdForOrientation(cube_cells) 
##    print (threshold)
    for j in range(0,len(cube_cells)-0):
        min_y, pos = minY(cube_cells, j)
##        print (j,"+",pos,"?","min_y", min_y)
        temp = cube_cells[j]
        del cube_cells[j]
        cube_cells.insert(j,cube_cells[pos-1])
        del cube_cells[pos]
        cube_cells.insert(pos,temp)

        min_x = None
        pos = 0
        for i in range (j, len(cube_cells)):
            xx = cube_cells[i][0][0][0]
            yy = cube_cells[i][0][0][1]
            if (yy > min_y - threshold and yy < min_y + threshold) and  (min_x == None or xx < min_x):
                min_x = xx
                pos = i
##        print (j,"+",pos,"??","min_x ",min_x)
        temp = cube_cells[j]
        del cube_cells[j]
        cube_cells.insert(j,cube_cells[pos-1])
        del cube_cells[pos]
        cube_cells.insert(pos,temp)

def origanlCoordinate(cube_cells, axis, pos, coord):
    """
    When using cv2.boundingRect it is giving the x and y axis among the smallest
    one of the square. So this fuction return the original coordinate of the
    edge point of the cube
    For eg:
    [[[ 96 226]] [[158 264]] [[118 329]] [[ 54 290]]]
    If coordinates of square is this, then cv2.boundingRect will give x and y
    as 54,226 but this function gives 54,290 which is required
    """
    for i in range (0,4):
        if cube_cells[pos][i][0][coord] == axis:
            return cube_cells[pos][i][0][1 - coord]
    

def angleToRotate(coord1, coord2):
    coord1_x = coord1[0][0]
    coord1_y = coord1[0][1]
    coord2_x = coord2[0][0]
    coord2_y = coord2[0][1]
    origin_x = coord1_x
    origin_y = coord2_y
    

    angle_minx = int (findAngle( (coord1_x, coord1_y), \
                                 (coord2_x, coord2_y), \
                                 (origin_x, origin_y)))
    angle_miny = int (findAngle( (coord2_x, coord2_y), \
                                 (coord1_x, coord1_y), \
                                 (origin_x, origin_y)))


##    print ("\n coord1_x",coord1_x )
##    print ("\n coord1_y", coord1_y)
##    print ("\n coord2_x",coord2_x )
##    print ("\n coord2_y",coord2_y )
##    print ("\n origin_x",origin_x )
##    print ("\n origin_y", origin_y)
##    print ("\n angle_minx",angle_minx )
##    print ("\n angle_miny", angle_miny)

    
    if coord2_y < coord1_y:
        return angle_minx if angle_minx < angle_miny else -angle_miny
    else:
        return -angle_minx if angle_minx < angle_miny else angle_miny

def trackingBox(cube_cells):
    g=0
    
def colorExtraction(cube_cells,image):
    img = image.copy()
    print (img)
    img1 = image.copy()
    height , width = 400,400
    mask = np.zeros((height,width), np.uint8)
    
    color = []
##    for sc in cube_cells:
    sc=cube_cells[0]
    (x, y, w, h) = cv2.boundingRect(sc)
    min_xy = w if w < h else h
    x_axis = x + int((w/2)) 
    y_axis = y + (int(h/2))
    cv2.circle(mask,(x_axis,y_axis),min_xy,(255,255,255),thickness=-1)
    masked_data = cv2.bitwise_and(img1, img1, mask=mask)
    _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])
    crop = masked_data[y:y+h,x:x+w]
    cv2.imshow("ee",img1)
##        color.append(img[y_axis,x_axis])
##        print (x_axis,y_axis)
##    orange_low = [ 23  68 159]
##    orange_high = [ 83 128 239]
##    red_low = 
##    red_high = 
##    blue_low = 
##    blue_high = 
##    green_low = 
##    green_high = 
##    black_low = 
##    black_high = 
##    yellow_low = 
##    yellow_high = 
##    print (color) 
    g=0

def orderingCoordinatesOfSquareContours(cube_cells):
    def order_points(cube_cells):
        xSorted = cube_cells[np.argsort(cube_cells[:, 0][: , 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 0][:, 1]), :]
        (tl, bl) = leftMost
        rightMost = rightMost[np.argsort(rightMost[:, 0][:, 1]), :]
        (tr, br) = rightMost
        return np.array([tl, tr, br, bl], dtype="int32")

    
    for (i, c) in enumerate(cube_cells):
        if i >= 0:
                rect = order_points(c)
                cube_cells[i][0][0] = rect[0][0]
                cube_cells[i][1][0] = rect[1][0]
                cube_cells[i][2][0] = rect[2][0]
                cube_cells[i][3][0] = rect[3][0]

def addingRemaingContours(cube_cells,image):
    floor_cc = math.sqrt(len(cube_cells))
    
    xcount = 1
    ycount = 1
    x_array = [cube_cells[0][0][0][0]]
    y_array = [cube_cells[0][0][0][1]]

    for i in range (0, len(cube_cells)):
##            (x, y) = cube_cells[i][0][0]
        x = cube_cells[i][0][0][0]
        y = cube_cells[i][0][0][1]
        print (x," . ", y)
    for i in range (1, len(cube_cells)):
        check = 0
        x = cube_cells[i][0][0][0]
        y = cube_cells[i][0][0][1]
        w = cube_cells[i][1][0][0] - x
        h = cube_cells[i][3][0][1] - y
        print (w , h)
        for j in range (0, len(x_array)):
            if x_array[j] + w >  x and x_array[j] - w <  x:
                check = 1
                break
        if check == 0:
            x_array.append(x)
            xcount = xcount + 1
        check = 0
        for j in range (0, len(y_array)):
            if y_array[j] + w >  y and y_array[j] - w <  y:
                check = 1
                break
        if check == 0:
            y_array.append(y)
            ycount = ycount + 1
    x_array.sort()
    y_array.sort()
    w = 40
    
    i = 1
    while i < len(x_array):
        test = int(x_array[i-1] + w + (w / 3))
        if test + w <= x_array[i] or test - w >= x_array[i]:
            x_array.append(test)
            xcount = xcount + 1
            x_array.sort()
        i = i + 1

    i = 1
    while i < len(y_array):
        test = int(y_array[i-1] + w + (w / 3))
        if test + w <= y_array[i] or test - w >= y_array[i]:
            y_array.append(test)
            ycount = ycount + 1
            y_array.sort()
        i = i + 1

    print ("\n x_array", x_array )
    print ("\n y_array", y_array )
    print ("\n xcount", xcount )
    print ("\n ycount", ycount )

    if xcount == ycount:
        ww = w - (w / 3)
        print(cube_cells)
        for i in range (0, len(x_array)):
            for j in range (0, len(y_array)):
                check = 0
                for k in range (0, len(cube_cells)):
                    if cube_cells[k][0][0][0] > x_array[i] - w and \
                       cube_cells[k][0][0][0] < x_array[i] + w and \
                       cube_cells[k][0][0][1] > y_array[j] - w and \
                       cube_cells[k][0][0][1] < y_array[j] + w:
                        check = 1
                        break
                if check == 0:
                    cube_cells.append(np.array([[[x_array[i], y_array[j]]], \
                                                [[x_array[i] + ww, y_array[j]]], \
                                                [[x_array[i] + ww, y_array[j] + ww]], \
                                                [[x_array[i], y_array[j] + ww]]],np.int32))
                
        print(cube_cells)
        for i in range (0, len(x_array)):
            for j in range (0, len(y_array)):
                cv2.circle(image,(x_array[i],y_array[j]), int(1), (255,255,255), 2)
        
        
##        cv2.circle(image,(319,168), int(1), (0,255,255), 2)
##        gggg = np.array([[[303,90]], [[363,90]], [[363,150]], [[303,150]]],np.int32)
##        cube_cells.append(gggg)
        
        


        
            
        print("asdasdasda")
    else:
        
        print("gg")
        

def myMain(image, state):
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
    dilated = cv2.dilate(canny, kernel, iterations = 2)
    ##cv2.imshow("3",dilated)

    """
    Finding Contours
    """
    (contours, hierarchy) = cv2.findContours(dilated.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ##cv2.drawContours(image, contours, -1, (255,0,0), 2)
    ##cv2.imshow("1",image)

    square_contours = squareApproximation (contours)
    if len(square_contours) < 1:
        return 0, image

    """
    Delete very small cotours in the beginning itself
    """
    i = 0
    while i < len(square_contours):
        if cv2.contourArea(square_contours[i]) < 50:
            del square_contours[i]
            i = i - 1
        i = i + 1
    
    square_in_square = searchSquareInSquare(square_contours)  
    cube_cells = eliminatingSquares(square_contours, square_in_square)
##    print(cube_cells)
    orderingCoordinatesOfSquareContours(cube_cells)
    
##    print("aa\n",cube_cells)
    if len(cube_cells) < 1:
        return 0, image
    if state == 0:
        min_x, pos1 = minX(cube_cells, 0)
        min_y, pos2 = minY(cube_cells, 0)
        distance = None
        pos = None
        for i in range (0, len(cube_cells)):
            d = math.sqrt( math.pow(cube_cells[i][0][0][0] - min_x, 2) + \
                           math.pow(cube_cells[i][0][0][1] - min_y, 2))
            if(distance == None or distance > d):
                distance = d
                pos = i
        angle = angleToRotate(cube_cells[pos][0], cube_cells[pos][1])
        print(angle, pos)
        if angle > 10 or angle < -10:
            (h, w) = image.shape[:2]
            M = cv2.getRotationMatrix2D((cube_cells[pos][0][0][0],\
                                         cube_cells[pos][0][0][1]), angle, 1)
            rotated = cv2.warpAffine(image, M, (w, h))
            path = os.path.dirname(os.path.realpath(__file__))
            path = path+"\\Images\\"
            cv2.imwrite(path+'Left.png', rotated)
            cv2.imshow("asdsada",rotated)
            print("x")
            return 1, rotated
        else:
            addingRemaingContours(cube_cells,image)
##    if len(cube_cells) > 4:
##        orientation(cube_cells)
##        angle = angleToRotate(cube_cells)
##        print (angle)
##        img = image.copy()
##         
##            rotateImage(cube_cells, img, angle)
##            
    ##print (cube_cells)
    ##    print (len(square_contours))
    ##    print (len(square_in_square))
    ##    print (len(cube_cells))
##    gggg = np.array([[[20,20]], [[120,20]], [[20,120]], [[120,120]]],np.int64)
####    gggg = gggg.astype('int32')
##    cube_cells.append(gggg)
##    print(cube_cells)
##    print(cube_cells[11].dtype)
##    
    cccc=0
    img = image.copy()
    for sc in cube_cells:
        x = sc[0][0][0]
        y = sc[0][0][1]
        w = sc[1][0][0] - x
        h = sc[3][0][1] - y
        min_xy = w if w < h else h
##        cv2.circle(img,(x + int((w/2)), y + (int(h/2))), int(min_xy/2), (255,255,255), 2)
        cv2.drawContours(img, [sc], -1, (0,255,0), 2)
        cv2.putText(img,str(cccc),cv2.boundingRect(sc)[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)
        cccc=cccc+1
            
##    print(cube_cells)
    cv2.imshow("5",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0, img

##
if __name__ == '__main__':
    image=cv2.imread("Left1.png")
    myMain(image, 0)

    
