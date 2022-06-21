# with norfair
from InfraRedRuntime import InfraRedRuntime
import pygame
import cv2
import numpy as np
import math
import norfair
from norfair import Detection, Tracker
import pyautogui
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

inputFrame = InfraRedRuntime()
image = ""
corners = 0
x_coord = []
y_coord = []
isChosen = False
secondHandDetected = False

# Distance function
def centroid_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


DETECTION_THRESHOLD = 0.6
DISTANCE_THRESHOLD = 0.8
KEYPOINT_DIST_THRESHOLD = 15


def keypoints_distance(detected_pose, tracked_pose):
    # Use different distances for bounding boxes and keypoints
    if detected_pose.label != 0:
        detection_centroid = np.sum(detected_pose.points, axis=0) / len(detected_pose.points)
        tracked_centroid = np.sum(tracked_pose.estimate, axis=0) / len(detected_pose.points)
        distances = np.linalg.norm(detection_centroid - tracked_centroid, axis=0)
        return distances / (KEYPOINT_DIST_THRESHOLD + distances)

    else:
        distances = np.linalg.norm(detected_pose.points - tracked_pose.estimate, axis=1)
        match_num = np.count_nonzero(
            (distances < KEYPOINT_DIST_THRESHOLD)
            * (detected_pose.scores > DETECTION_THRESHOLD)
            * (tracked_pose.last_detection.scores > DETECTION_THRESHOLD)
        )
        return 1 / (1 + match_num)


tracker = Tracker(distance_function=keypoints_distance, distance_threshold=DISTANCE_THRESHOLD,
                  detection_threshold=DETECTION_THRESHOLD)


# while croping the table from whole video
def click_event(event, x, y, flags, params):
    global image, corners, isChosen
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        corners = corners + 1
        print(x, ' ', y)
        x_coord.append(x)
        y_coord.append(y)

    if corners == 4:
        isChosen = True


def bwareaopen(img):
    """Remove small objects from binary image (approximation of
    bwareaopen in Matlab for 2D images).

    Args:
    img: a binary image (dtype=uint8) to remove small objects from
    min_size: minimum size (in pixels) for an object to remain in the image
    connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).

    Returns:
        the binary image with small objects removed
     """

    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 1500

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2


# we delete the farest far point in the farpoints, to remove the farpoint occuring on the arm
def checkFarPoints(farPoints):
    y_coords = []
    # get the y coordinates of all far points and sort them
    for i in farPoints:
        y_coords.append(i[1])
    # y_coords=y_coords.sort()
    tobeDeletedY = 1000000  # assigning a random value only for requirement
    if y_coords is not None and len(
            y_coords) >= 2:  # calcualte the difference between first 2 elements and last 2 elements , then decide which one is the farest element
        firstDiff = y_coords[-1] - y_coords[-2]  # last 2
        secondDiff = y_coords[1] - y_coords[0]  # first 2
        if firstDiff > secondDiff:
            tobeDeletedY = y_coords[-1]
        elif firstDiff < secondDiff:
            tobeDeletedY = y_coords[0]
    updFarPoints = []
    for i in farPoints:
        if i[1] != tobeDeletedY:
            updFarPoints.append(i)
    return updFarPoints


def falseFarPointLocation(farPoints):
    y_coords = []
    # get the y coordinates of all far points and sort them
    for i in farPoints:
        y_coords.append(i[1])
    # y_coords=y_coords.sort()
    tobeDeletedY = 1000000  # assigning a random value only for requirement
    tobeDeletedX = 1000000
    if y_coords is not None and len(
            y_coords) >= 2:  # calcualte the difference between first 2 elements and last 2 elements , then decide which one is the farest element
        firstDiff = y_coords[-1] - y_coords[-2]  # last 2
        secondDiff = y_coords[1] - y_coords[0]  # first 2
        if firstDiff > secondDiff:
            tobeDeletedY = y_coords[-1]
            tobeDeletedX = farPoints[len(farPoints) - 1][0]
        elif firstDiff <= secondDiff:
            tobeDeletedY = y_coords[0]
            tobeDeletedX = farPoints[0][0]
    return tobeDeletedY, tobeDeletedX


def chooseBoundsFar(farPoints):
    if len(farPoints) >= 2:
        farPoints = checkFarPoints(farPoints)
    x_sum = 0
    y_sum = 0
    for i in range(len(farPoints)):
        x_sum += farPoints[i][0]
        y_sum += farPoints[i][1]
    x_avg = x_sum / len(farPoints)
    y_avg = y_sum / len(farPoints)
    coord = (x_avg, y_avg)
    return coord


def choosePeakPoint(peakPoints):
    # decide if hands lay on x line or y line
    x_coords = []
    y_coords = []
    chosen_x_coordinate = 10 ** 20
    chosen_y_coordinate = 10 ** 20
    lay_x_horizontal = True

    # get the y coordinates of all peak points and sort them
    for x, y in peakPoints:
        # y_coords.append(y)
        y_coords += y
    y_coords = y_coords.sort()
    print("y_coords", y_coords)
    # get the x coordinates of all peak points and sort them
    for x, y in peakPoints:
        # x_coords.append(x)
        x_coords += x
    x_coords = x_coords.sort()
    result_point = peakPoints[0]
    print("EVETTTT  ", y_coords)
    print("noooooo  ", x_coords)

    if y_coords is not None and len(y_coords) >= 2:
        print("hello")
        x_minmax_dif = x_coords[-1] - x_coords[0]
        y_minmax_dif = y_coords[-1] - y_coords[0]
        if (x_minmax_dif <= y_minmax_dif):
            lay_x_horizontal = False
            print("1")

        # decide the peak blue point which we willo use as center of bounding box
        # decide if max point or min point will be taken
        if (lay_x_horizontal == True):
            print("2")
            if (x_coords[len(x_coords / 2)] <= x_minmax_dif / 2 + x_coords[0]):
                chosen_x_coordinate = x_coords[0]
                print("3")
            elif (x_coords[len(x_coords / 2)] > x_minmax_dif / 2 + x_coords[0]):
                chosen_x_coordinate = x_coords[-1]
                print("4")
        elif (lay_x_horizontal == False):
            print("1")  # y koordinatınca uzanıyor kol
            if (y_coords[len(y_coords / 2)] <= y_minmax_dif / 2 + y_coords[0]):
                chosen_y_coordinate = y_coords[0]
                print("5")
            elif (y_coords[len(y_coords / 2)] > y_minmax_dif / 2 + y_coords[0]):
                chosen_y_coordinate = y_coords[-1]
                print("6")

        # x koordiantı mı y koorinati mı bilinioyr ona göre noktayı bulucaz
        if (chosen_x_coordinate != 10 ** 20):
            print("7")  # x koordiantı belirlenmiştir
            for i in peakPoints:
                print("8")
                if i[0] == chosen_x_coordinate:
                    print("9")
                    result_point = i
        else:  # y kordinatı belirlenmiştir
            print("10")
            for i in peakPoints:
                print("11")
                if i[1] == chosen_y_coordinate:
                    print("12")
                    result_point = i
    # print("result point for bbox is : ", result_point)
    return result_point


def learnDirection(peakPoints, maxY, minY, maxX, minX,crop_img):
    height = crop_img.shape[0]
    width = crop_img.shape[1]
    for i in range(height):
        if crop_img[i][0]==255:  #left of the table
            #print("white-left")
            return 3
        elif crop_img[i][width-1]==255: #right
            #print("white-right")
            return 4
    for j in range(width):
        if crop_img[0][j]==255:  #down
            #print("white-down")
            return 1
        elif crop_img[height-1][j]==255: #up
            #print("white-up")
            return 2
    return 0

def eliminatePoints(minY, maxY, minX, maxX, allvec, peakPoints):
    updPeakPoints = []
    allvec2D = np.reshape(allvec, (maxY - minY, maxX - minX))
    #dir = learnDirection(peakPoints, maxY, minY, maxX, minX)
    for i in range(len(peakPoints)):
        x_tmp = peakPoints[i][0]
        y_tmp = peakPoints[i][1]
        x_correspond = int(((maxX - minX) / 800) * x_tmp)
        y_correspond = int(((maxY - minY) / 600) * y_tmp)
        if y_correspond < maxY - minY and y_correspond >= 0:
            if x_correspond < maxX - minX and x_correspond >= 0:

                boundary_y_up = y_correspond + 30
                boundary_y_down = y_correspond - 30
                boundary_x_up = x_correspond + 30
                boundary_x_down = x_correspond - 30

                if y_correspond + 30 > maxY - minY:
                    boundary_y_up = maxY - minY - 1
                elif y_correspond - 30 < 0:
                    boundary_y_down = 0
                if x_correspond + 30 > maxX - minX:
                    boundary_x_up = maxX - minX - 1
                elif x_correspond - 30 < 0:
                    boundary_x_down = 0
                temp_list = []
                for z in range(boundary_x_down, boundary_x_up):
                    for j in range(boundary_y_down, boundary_y_up):
                        if allvec2D[j][z] != 0:
                            temp_list.append(allvec2D[j][z])
                #print("tmp list min", min(temp_list))
                if min(temp_list) >= 2300:
                    tmp_list = []
                    tmp_list.append(peakPoints[i][0])
                    tmp_list.append(peakPoints[i][1])
                    updPeakPoints.append(tmp_list)
            else:
                tmp_list = []
                tmp_list.append(peakPoints[i][0])
                tmp_list.append(peakPoints[i][1])
                updPeakPoints.append(tmp_list)
        else:
            tmp_list = []
            tmp_list.append(peakPoints[i][0])
            tmp_list.append(peakPoints[i][1])
            updPeakPoints.append(tmp_list)
    #print(len(updPeakPoints))
    #print("-----")
    return updPeakPoints
def choosePoint(peakPoints, farPoints, maxY, minY, maxX, minX, infTable, allvec, direction, crop_img):
    resultPoint=[]
    #print("far point amount:", len(farPoints))
    if peakPoints is not None and len(peakPoints) != 0:
        direction = learnDirection(peakPoints, maxY, minY, maxX, minX, crop_img)
        print("direction",direction)
        if direction==2:
            minVal = 1000
            idx = 0
            for i in range(len(peakPoints)):
                if peakPoints[i][1] < minVal:
                    idx = i
                    minVal = peakPoints[i][1]
            y_point = peakPoints[idx][1]
            x_point = peakPoints[idx][0]
            print(y_point)
            print(x_point)
        elif direction==1:
            maxVal = 0
            idx = 0
            print(peakPoints)
            for i in range(len(peakPoints)):
                if peakPoints[i][1] >= maxVal:
                    idx = i
                    maxVal = peakPoints[i][1]
            y_point = peakPoints[idx][1]
            x_point = peakPoints[idx][0]
            print(y_point)
            print(x_point)
        elif direction==3:
            maxVal=0
            idx=0
            for i in range(len(peakPoints)):
                if peakPoints[i][0]>=maxVal:
                    idx=i
                    maxVal=peakPoints[i][0]
            y_point = peakPoints[idx][1]
            x_point = peakPoints[idx][0]
        elif direction==4:
            minVal = 1000
            idx = 0
            for i in range(len(peakPoints)):
                if peakPoints[i][0] < minVal:
                    idx = i
                    minVal = peakPoints[i][0]
            y_point = peakPoints[idx][1]
            x_point = peakPoints[idx][0]
        else:
            y_point = peakPoints[0][1]
            x_point = peakPoints[0][0]

        resultPoint.append(x_point)
        resultPoint.append(y_point)
        #print(resultPoint)
    return resultPoint, direction


def Average(lst):
    return sum(lst) / len(lst)


def checkExistence(x_sec, y_sec, w_sec, h_sec, minY, maxY, minX, maxX, allvec, str_info):
    allvec2D = np.reshape(allvec, (maxY - minY, maxX - minX))
    # print(min(allvec))
    bb_vec = []
    for i in range(x_sec, (x_sec + w_sec)):
        for j in range(y_sec, (y_sec + h_sec)):
            x_correspond = int(((maxX - minX) / 800) * i)
            y_correspond = int(((maxY - minY) / 600) * j)
            bb_vec.append(allvec2D[y_correspond][x_correspond])

    if (Average(bb_vec) <= 2400):
        return False
    return True

def run():
    global image, isChosen, x_coord, y_coord, isTracking, tracker
    isClicked = False
    kinectDepth = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
    kinect = inputFrame.getkinect()
    # -------- Main Program Loop -----------
    # variables for click operation
    count = 0
    default_x = 200
    default_y = 200
    prev_x_diff = 0
    prev_y_diff = 0
    while not inputFrame.getdone():
        # --- Main event loop
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                inputFrame.setdone(True)  # Flag that we are done so we exit this loop

            elif event.type == pygame.VIDEORESIZE:  # window resized
                inputFrame.setscreen(pygame.display.set_mode(event.dict['size'],
                                                             pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE,
                                                             32))

        # --- Getting frames and drawing
        if kinect.has_new_infrared_frame():
            frame = kinect.get_last_infrared_frame()
            inputFrame.draw_infrared_frame(frame, inputFrame.getframe_surface())
            frame = None
        frameD = None
        frameDepth = None
        if kinectDepth.has_new_depth_frame():
            frameDepth = kinectDepth.get_last_depth_frame()
            frameD = kinectDepth._depth_frame_data
            # print(frameD)
            frameDepth = frameDepth.astype(np.uint8)

            frameDepth = np.reshape(frameDepth, (424, 512))
            frameDepth = cv2.cvtColor(frameDepth, cv2.COLOR_GRAY2RGB)
            # cv2.imshow('KINECT Video Stream', frameDepth)
        screen = inputFrame.getscreen()
        screen.blit(inputFrame.getframe_surface(), (0, 0))
        pygame.image.save(screen, "screenshot.jpeg")
        pygame.display.update()
        image = cv2.imread("screenshot.jpeg")
        cv2.imshow("cropped", image)

        if isChosen:

            tableCrop = image[min(y_coord):max(y_coord), min(x_coord):max(x_coord)]
            img = cv2.cvtColor(tableCrop, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(img, 65, 255, cv2.THRESH_BINARY_INV)
            if frameDepth is not None:
                tableCropD = frameDepth[min(y_coord):max(y_coord), min(x_coord):max(x_coord)]
                #depth image
                cv2.imshow('KINECT Video Stream', tableCropD)
            allvec = []
            for i in range(min(y_coord), max(y_coord)):
                for j in range(min(x_coord), max(x_coord)):
                    if frameD is not None:
                        Pixel_Depth = frameD[((i * 512) + j)]
                        allvec.append(Pixel_Depth)

            # the window showing output image
            dsize = (800, 600)
            # resize image
            output = cv2.resize(thresh1, dsize)
            output = bwareaopen(output)

            crop_img = output.astype(np.uint8)
            contours, hierarchy = cv2.findContours(crop_img.copy(), cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_NONE)  # finding contour
            # original infrared image

            # resize image
            infTable = cv2.resize(tableCrop, dsize)

            # find contour with max area
            if len(contours) != 0:
                contour_list = []
                cnt_second = []
                # find largest area contour
                cnt_list = sorted(contours, key=cv2.contourArea)
                cnt = cnt_list[-1]
                if len(cnt_list) >= 2:
                    cnt_second = cnt_list[-2]
                # cnt = max(contours, key=lambda x: cv2.contourArea(x))
                x_sec, y_sec, w_sec, h_sec = 0, 0, 0, 0
                x, y, w, h = cv2.boundingRect(cnt)
                bothHand = 0
                if len(cnt_second) != 0:
                    area = cv2.contourArea(cnt_second)
                    # print("area",area)
                    if area > 900:
                        x_sec, y_sec, w_sec, h_sec = cv2.boundingRect(cnt_second)
                        if (checkExistence(x, y, w, h, min(y_coord), max(y_coord), min(x_coord),
                                           max(x_coord), allvec, "first")):
                            cv2.rectangle(infTable, (x, y), (x + w, y + h), (94, 253, 138), 0)
                            bothHand += 1
                        # second contour rectangle
                        if (checkExistence(x_sec, y_sec, w_sec, h_sec, min(y_coord), max(y_coord), min(x_coord),
                                           max(x_coord), allvec, "second")):
                            cv2.rectangle(infTable, (x_sec, y_sec), (x_sec + w_sec, y_sec + h_sec), (148, 77, 170), 0)
                            bothHand += 1
                            cnt = cnt_second


                        if bothHand == 2:
                            secondHandDetected = True
                            print("two boxs detected")
                        else:
                            secondHandDetected = False
                    else:
                        secondHandDetected = False
                else:
                    cv2.rectangle(infTable, (x, y), (x + w, y + h), (94, 253, 138), 0)
                    secondHandDetected = False

                if secondHandDetected == False:
                    # finding convex hull
                    hull = cv2.convexHull(cnt)

                    # drawing contours
                    drawing = np.zeros(infTable.shape, np.uint8)
                    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
                    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

                    # finding convex hull
                    hull = cv2.convexHull(cnt, returnPoints=False)

                    # finding convexity defects
                    defects = cv2.convexityDefects(cnt, hull)
                    count_defects = 0
                    cv2.drawContours(infTable, contours, -1, (0, 255, 0), 3)

                    # applying Cosine Rule to find angle for all defects (between fingers)
                    # with angle > 90 degrees and ignore defects
                    peakPoints = []
                    farPoints = []
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]

                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])

                        # find length of all sides of triangle
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                        # apply cosine rule here

                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                        # ignore angles > 90 and highlight rest with red dots
                        if angle <= 90:
                            count_defects += 1
                            cv2.circle(infTable, far, 10, [0, 0, 255], -1)
                            farPoints.append(far)
                        # dist = cv2.pointPolygonTest(cnt,far,True)
                        # draw a line from start to end i.e. the convex points (finger tips)
                        # (can skip this part)
                        cv2.line(infTable, start, end, [0, 255, 0], 2)
                        cv2.circle(infTable, start, 5, [255, 0, 0], -1)
                        peakPoints.append(start)
                    # print(farPoints)

                    # to track hand by bounding box
                    # first, we do selections of farpoints
                    fromBelow = False
                    fromLeft = False
                    direction = 1
                    if len(peakPoints) != 0:
                        # print("FAR POINTS   ", farPoints)
                        centerOfBBox, direction= choosePoint(peakPoints, farPoints, max(y_coord), min(y_coord),
                                                               max(x_coord), min(x_coord), infTable, allvec,direction,crop_img)
                        #print("center of bbox:", centerOfBBox)
                        if len(centerOfBBox) == 2:
                            x = int(centerOfBBox[0])
                            y = int(centerOfBBox[1])

                    w = 35
                    h = 35
                    #for click operation, check if mouse moves or not
                    if abs(x - default_x) < 5 or abs(y - default_y) < 5:
                         x = default_x
                         y = default_y
                         count = count + 1
                    else:
                         count = 0
                         default_x = x
                         default_y = y

                    # for panning 1
                    ismousedown = False
                    if len(farPoints) >= 3 and count == 5:
                        pyautogui.mouseDown()
                        ismousedown = True
                        count = 0
                    if ismousedown == True and count == 5:
                        pyautogui.mouseUp()

                    cv2.rectangle(infTable, (x , y), (x + w, y + h), (128, 255, 0), 2)

                    bbox = np.array(
                        [
                            [x - w, y - h],
                            [w + x, h + y]
                        ]
                    )
                    detections = []
                    detections.append(
                        Detection(points=bbox)
                    )
                    tracked_objects = tracker.update(detections=detections)
                    id_draw_position = norfair.draw_tracked_objects(infTable, tracked_objects, color=(255, 0, 255),
                                                                    id_size=2)

                    if id_draw_position is not None and len(
                            id_draw_position) == 2:  # we arange the mouse position by position of bounding box

                        MouseX = -400
                        MouseY = 300
                        if direction == 1:
                            MouseX = (-1) * (((-1) * id_draw_position[0]) + 800)
                            MouseY = (((-1) * id_draw_position[1]) + 600) - 60
                        elif direction == 2:
                            MouseX = (-1) * (((-1) * id_draw_position[0]) + 800)
                            MouseY = (((-1) * id_draw_position[1]) + 600) + 30
                        elif direction == 3:
                            MouseX = (-1) * (((-1) * id_draw_position[0]) + 800) + 30
                            MouseY = (((-1) * id_draw_position[1]) + 600)
                        elif direction == 4:
                            MouseX = (-1) * (((-1) * id_draw_position[0]) + 800) - 45
                            MouseY = (((-1) * id_draw_position[1]) + 600)
                        pyautogui.moveTo(MouseX, MouseY)

                    # CLICK
                    # for click operation
                    if count == 15:
                        pyautogui.click()
                        count = 0

                else:
                    x_diff_coord = abs(x_sec - x)
                    y_diff_coord = abs(y_sec - y)

                    if prev_x_diff == 0 and prev_y_diff == 0:
                        prev_x_diff = x_diff_coord
                        prev_y_diff = y_diff_coord
                    else:
                        if (prev_x_diff > x_diff_coord and prev_x_diff - x_diff_coord >= 7) or (
                                prev_y_diff > y_diff_coord and prev_y_diff - y_diff_coord >= 7):  # zoom out
                            pyautogui.scroll(-20)
                            print("ZOOM OUT")

                        elif (prev_y_diff < y_diff_coord and y_diff_coord - prev_y_diff >= 7) or (
                                prev_x_diff < x_diff_coord and x_diff_coord - prev_x_diff >= 7):  # zoom in
                            pyautogui.scroll(20)
                            print("ZOOM IN")

                        prev_x_diff = x_diff_coord
                        prev_y_diff = y_diff_coord

            cv2.imshow('table', crop_img)
            cv2.imshow('tableInf', infTable)
        else:
            cv2.setMouseCallback('cropped', click_event)

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # --- Limit to 60 frames per second
        clk = inputFrame.getclock()
        clk.tick(15)

    # end of main program loop (while loop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Close our Kinect sensor, close the window and quit.
    kinect.close()
    pygame.quit()


run()
