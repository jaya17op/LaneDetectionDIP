import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import glob
import argparse
import os

class LaneDetection:

    def __init__(self):
        self.best_left_fit = np.array([1,1,1])
        self.best_right_fit = np.array([1,1,1])

    # function to read the camera parameters
    def readCameraParameters(self, dataset):
        K = np.array([[9.037596e+02, 0.00000000e+00, 6.957519e+02], [0.00000000e+00, 9.019653e+02, 2.242509e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        D = np.array([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]])

        if (dataset == '2'):
            K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            D = np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])
        return K, D

    # function to undistort the image
    def undistortImage(self, image, K, D):
        return cv2.undistort(image,K,D,None,K)

    # function to get the warped image
    def getWarpedImage(self, image, src, dst):
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, H, (300, 300))
        return warped, H

    # function to do HLS color thresholding
    def doColorThresholding(self, warped):
        #https://www.w3schools.com/colors/colors_hsl.asp

        hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS).astype(float)

        #Seperate yellow
        lower_yellow = np.array([20,90,55],dtype=np.uint8)
        upper_yellow = np.array([45,200,255],dtype=np.uint8)
        yellow_mask = cv2.inRange(hls,lower_yellow,upper_yellow)
        yellow_line = cv2.bitwise_and(hls, hls, mask=yellow_mask).astype(np.uint8)

        #Seperate White
        lower_white = np.array([0,200,0],dtype=np.uint8)
        upper_white = np.array([255,255,255],dtype = np.uint8)
        white_mask = cv2.inRange(hls,lower_white,upper_white)
        white_line = cv2.bitwise_and(hls, hls, mask=white_mask).astype(np.uint8)

        mask = cv2.bitwise_or(white_mask,yellow_mask)
        preprocessed_hls = cv2.bitwise_or(yellow_line, white_line)
        preprocessed_img = cv2.bitwise_and(warped, warped, mask=mask).astype(np.uint8)

        return mask

    # function to generate histogram
    def generateHistogram(self, mask):
        hist = np.sum(mask, axis=0)
        midpoint = hist.shape[0]//2
        left_lane_ix = int(np.argmax(hist[:midpoint]))
        right_lane_ix = np.argmax(hist[midpoint:]) + midpoint
        return hist, left_lane_ix, right_lane_ix

    # function to fit a polynomial
    def fitPolynomial(self, mask, left_lane_list, right_lane_list, ones_x, ones_y, left_lane_x, left_lane_y, right_lane_x, right_lane_y):
        margin = 5
        try:
            left_fit = np.polyfit(left_lane_y, left_lane_x, 2)
            right_fit = np.polyfit(right_lane_y, right_lane_x, 2)
            # check for good coefficients and save this if good configuration
            if(self.best_left_fit is not None and self.best_right_fit is not None):
                if (abs(left_fit[1]-self.best_left_fit[1]) > 1.5):
                    #print('Take the last well known left fit')
                    left_fit = self.best_left_fit
                if (abs(right_fit[1]-self.best_left_fit[1]) > 1.5):
                    #print('Take the last well known right fit')
                    right_fit = self.best_right_fit
        except:
            #print('Error in fitting polynomial ; think of some method to solve and fit this')
            left_fit, right_fit = self.best_left_fit, self.best_right_fit
            #print(left_fit, right_fit)

        self.best_left_fit, self.best_right_fit = left_fit, right_fit

        # polynomial equation
        point_y = np.linspace(0, mask.shape[0]-1, mask.shape[0])
        left_line_x = left_fit[0]*(point_y**2) + left_fit[1]*(point_y) + left_fit[2]
        right_line_x = right_fit[0]*(point_y**2) + right_fit[1]*(point_y) + right_fit[2]

        # center line equation
        center_line_x = (left_line_x+right_line_x)/2
        center_fit = np.polyfit(point_y, center_line_x, 1)
        slope_center = center_fit[1]

        out_img = np.dstack((mask, mask, mask))
        window_img = np.zeros_like(out_img)

        # color mofication of non-zero pixels
        out_img[ones_y[left_lane_list], ones_x[left_lane_list]] = [255, 0, 0]
        out_img[ones_y[right_lane_list], ones_x[right_lane_list]] = [0, 255, 0]

        # Stack individual lane points
        left_line_window1 = np.array([np.transpose(np.vstack([left_line_x-margin, point_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_line_x+margin, point_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_line_x-margin, point_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_line_x+margin, point_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0,0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255,0))
        result = cv2.addWeighted(out_img, 1, window_img, 1, 0)

        # Stack the lane points together
        pts_left = np.array([np.transpose(np.vstack([left_line_x, point_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line_x, point_y])))])
        pts = np.hstack((pts_left, pts_right))

        #draw center line
        pts_center = np.array([np.transpose(np.vstack([center_line_x, point_y]))])
        cv2.polylines(result, np.int32([pts_center]), isClosed=False, color=(102,2,10), thickness=2)

        # Fill the lane polynomial
        cv2.fillPoly(result, np.int_([pts]),(0,0,255))

        return result, point_y, left_fit, right_fit, slope_center

    # function to predict turn
    def predict_turn(self, center_line_slope):
        if (center_line_slope > 160.0):
            return 'Prediction: Right Turn'
        if (center_line_slope >= 130.0 and center_line_slope <= 160.0):
            return 'Prediction: Go straight'
        if (center_line_slope < 130.0):
            return 'Prediction: Left Turn'

    # function to get the lanes
    def getLanes(self, mask, left_lane_ix, right_lane_ix):
        sliding = np.dstack((mask, mask, mask))
        margin = 30
        windows = 10
        min_pix = 50
        mean_left = left_lane_ix
        mean_right = right_lane_ix

        window_height = int(mask.shape[0]/windows)

        # calculate the indices of non zero pixels
        ones_in_mask = mask.nonzero()
        ones_y = np.array(ones_in_mask[0])
        ones_x = np.array(ones_in_mask[1])

        # left lane
        left_border_left = mean_left - margin
        left_border_right = mean_left + margin

        # right lane
        right_border_left = mean_right - margin
        right_border_right = mean_right + margin

        left_lane_list = []
        right_lane_list = []

        for window in range(windows):
            # top and botton margin
            top_border = mask.shape[0] - (window)*window_height
            bottom_border = mask.shape[0] - (window+1)*window_height

            # check if the windows has the "ones" pixels, if yes add those to pixel list
            left_lane = (((ones_y >= bottom_border) & (ones_y < top_border) & 
                    (ones_x >= left_border_left) & (ones_x < left_border_right)).nonzero())[0]

            right_lane = (((ones_y >= bottom_border) & (ones_y < top_border) & 
                    (ones_x >= right_border_left) & (ones_x < right_border_right)).nonzero())[0]

            left_lane_list.append(left_lane)
            right_lane_list.append(right_lane)

            # if the left and right lane pixels are above threshold, then re-center the window and update the borders
            if (len(left_lane) > min_pix):
                mean_left = int(np.mean(ones_x[left_lane]))
                left_border_left = mean_left - margin
                left_border_right = mean_left + margin
                #print('update left lane mean to ' + str(mean_left))

            if (len(right_lane) > min_pix):
                mean_right = int(np.mean(ones_x[right_lane]))
                right_border_left = mean_right - margin
                right_border_right = mean_right + margin
                #print('update right lane mean to ' + str(mean_right))

            # draw the window
            cv2.rectangle(sliding,(left_border_left,bottom_border),(left_border_right,top_border),(0,255,0), 1) 
            cv2.rectangle(sliding,(right_border_left,bottom_border),(right_border_right,top_border),(0,0,255), 1)

        left_lane_list = np.concatenate(left_lane_list)
        right_lane_list = np.concatenate(right_lane_list)

        # store the left and right lane Non-zero(white) pixels
        left_lane_x = ones_x[left_lane_list]
        left_lane_y = ones_y[left_lane_list]

        right_lane_x = ones_x[right_lane_list]
        right_lane_y = ones_y[right_lane_list]

        result, point_y, left_fit, right_fit, slope_center = self.fitPolynomial(mask, left_lane_list, right_lane_list, ones_x, ones_y, left_lane_x, left_lane_y, right_lane_x, right_lane_y)

        turning = self.predict_turn(slope_center)

        return result, sliding, turning


def dataSet2(args):
    lane = LaneDetection()
    src_pts = np.float32([(200,580),(600,395),(740,395),(1080,580)])
    dst_pts = np.float32([(0,300),(0,0),(300,0),(300,300)])

    file = args['full_path']
    dataset = args['set']
    K, D = lane.readCameraParameters(dataset)
    cap = cv2.VideoCapture(file)

    video = cv2.VideoWriter('Lane_Detection_video_challenge.avi',cv2.VideoWriter_fourcc(*'XVID'), 20,(1280,600))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # undistort the image
            frame = cv2.undistort(frame,K,D,None,K)
            image = cv2.resize(frame, (1280, 600))
            # warp the image
            warped, H = lane.getWarpedImage(image, src_pts, dst_pts)
            # get the color threshold mask
            mask = lane.doColorThresholding(warped)
            # get the histogram of lane pixels
            hist, left_lane_ix, right_lane_ix = lane.generateHistogram(mask)
            # get the lanes
            result, sliding, turning = lane.getLanes(mask, left_lane_ix, right_lane_ix)

            size = (image.shape[:2][1], image.shape[:2][0])
            # reverse warp and meld the images
            reverse_warp = cv2.warpPerspective(result, np.linalg.inv(H), size)
            final = cv2.addWeighted(image, 1, reverse_warp, 1, 0)
            #final[:300,980:,:] = sliding

            cv2.putText(final, turning, (450, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

            cv2.imshow('Final', final)

            video.write(final)

            if cv2.waitKey(25) & 0XFF == ord('q'):
                break
        else:
            break

    cv2.destroyAllWindows()

    video.release()

def dataSet1(args):
    lane = LaneDetection()
    src_pts = np.float32([(120,499),(500,290),(700,290),(950,499)])
    dst_pts = np.float32([(0,300),(0,0),(300,0),(300,300)])

    video = cv2.VideoWriter('LaneDetection_DataSet_1.avi',cv2.VideoWriter_fourcc(*'XVID'), 10,(1280,500))
    folder = args['full_path']
    if (not folder[-1] == '/'):
        folder = folder + '/'
    images = glob.glob(folder +'*')
    images.sort()

    dataset = args['set']
    K, D = lane.readCameraParameters(dataset)

    for path in images:
            image = cv2.imread(path)
            # undistort the image
            image = cv2.undistort(image,K,D,None,K)
            image = cv2.resize(image, (1280, 500))
            # warp the image
            warped, H = lane.getWarpedImage(image, src_pts, dst_pts)
            # get the color threshold mask
            mask = lane.doColorThresholding(warped)
            # get the histogram of lane pixels
            hist, left_lane_ix, right_lane_ix = lane.generateHistogram(mask)
            # get the lanes
            result, sliding, turning = lane.getLanes(mask, left_lane_ix, right_lane_ix)

            size = (image.shape[:2][1], image.shape[:2][0])
            # reverse warp and meld the images
            reverse_warp = cv2.warpPerspective(result, np.linalg.inv(H), size)
            final = cv2.addWeighted(image, 1, reverse_warp, 1, 0)
            #final[:300,980:,:] = sliding

            cv2.putText(final, turning, (450, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

            cv2.imshow('Final', final)
            video.write(final)

            if cv2.waitKey(25) & 0XFF == ord('q'):
                break

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--set", required=False, help="Input 1 for Regular video OR 2 for Challenge video", default='2', type=str)
    parser.add_argument("-path", "--full_path", required=False, help="working directory of image data set or video", default='challenge_video.mp4', type=str)
    args = vars(parser.parse_args())

    if (not os.path.exists(args['full_path'])):
        print('Path does not exist ; Re run and enter correct path as per README')
        exit()

    if (args['set'] == '1'):
        dataSet1(args)
    elif(args['set'] == '2'):
        dataSet2(args)

