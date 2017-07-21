#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>

#include "common/defs.hpp"
#include "common/config.hpp"
#include "vision/config.hpp"
#include "vision/vision.hpp"
#include "vision/buoys_vision.hpp"
#include "image/image.hpp"

void vision_func(int* img, double* z_rotation, double* radius,  double* elevation, double* distance, double* confidence)
{
	cv::Mat image = imageRead(img);
	
	double conf = 0;

    double object_probability = 0;

	//0 100 200        
	//170 255 255
        
	int blue = 0;
    cv::Scalar lower(0,100,200);
    cv::Scalar upper(185, 230, 255);
        
	cv::Mat mask;
    cv::inRange(image, lower, upper, mask);
    while (cv::countNonZero(mask) < 700 and blue < 245)
	{
        cv::Scalar lower_high(0,100, 200);
        cv::Scalar upper_high(blue, 255, 255);
        cv::inRange(image, lower_high, upper_high, mask);
	}   


    if (blue > 190)
    {
        object_probability = object_probability+5;
        blue += 5;
    }
    else
    {
        blue += 10;
    }
    // finds pixels that are more orange than blue
    //if less than 700 pixels are found, then the requirements are loosened
    int n = 1;
    if (blue > 210)
    {
        n = (int) (object_probability/15);
    }
    
    cv::Mat kernel(2, 1, cv::CV_8UC1, cv::Scalar(1));
    cv::Mat dilation;
    cv::dilate(mask, dilation, kernel, n);
    // dialates pixels to make finding large contours easier

    cv::Mat ret;
    double thresh = cv::threshold(dilation, ret, 127, 255, 0);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::CV_RETR_TREE, cv::CV_CHAIN_APPROX_SIMPLE)
    
    double largestArea = -1;
    int largestIndex = -1;
    
    for (int i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        if (area > largestArea)
        {
            largestArea = area;
            largestIndex = i;
        }
    }
    
    if (largestIndex < 0)
    {
        return;
    }
    
    int white_pixels = cv::countNonZero(dilation);
    cv::cvtColor(dilation, dilation, cv::CV_GRAY2BGR);

    std::vector<cv::Point> cnt = contours[largestIndex];

    cv::RotatedRect rect = cv::minAreaRect(cnt);
    std::vector<cv::Point> box;
    cv::boxPoints(rect, box);
    cv::drawContours(dilation,box,0,cv::Scalar(0,255,0),2)
#draws box around largest contour



    double l1 = (box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2 
    double l2 = (box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2
        
    m = 0

#inverted y values because down is +y
    if (l1 > l2):
        if box[0][0]-box[1][0] == 0:
            m = 'undef' 
        else:
            m = -(box[0][1] - box[1][1]) / (box[0][0] - box[1][0])
    else:
        if box[1][0]-box[2][0] == 0:
            m = 'undef' 
        else:
            m = -(box[1][1] - box[2][1]) / (box[1][0] - box[2][0])
    if m != 'undef':
        theta = math.degrees(math.atan(m))
    else:
        theta = 90
    if (theta < 0):
        theta += 180
    bearing = 90 - theta
    
#        print(bearing)
#finds angle of path
    
    if l1 > l2:
        r = math.sqrt(l1)
    else:
        r = math.sqrt(l2)


    x_center = int((box[0][0]+box[1][0]+box[2][0]+box[3][0])/4)
    y_center = int((box[0][1] + box[1][1] + box[2][1] + box[3][1])/4)

    cv2.circle(dilation, (int(x_center), int(y_center)), 2, (255,125,125), thickness = 6, lineType = 8)
    cv2.circle(img, (int(x_center), int(y_center)), 2, (255,125,125), thickness = 6, lineType = 8)
#finds center of path



    length = 100
    x2 =  int(x_center + length * math.cos(theta * math.pi / 180.0))
    y2 =  int(y_center - length * math.sin(theta * math.pi / 180.0))
    
    cv2.line(dilation,(x_center, y_center), (x2, y2), (100, 100, 255), thickness = 4)
    cv2.line(img,(x_center, y_center), (x2, y2), (100, 100, 255), thickness = 4)
#draws line in direction of path
    

    draw =  np.copy(dilation) 
    cv2.drawContours(draw, contours, largestIndex,(0, 0, 0))
    #cv2.imshow("one_contour",draw)
        
    contour_area = cv2.contourArea(contours[largestIndex])
    #print(contour_area)

    #print(math.sqrt(l1)*math.sqrt(l2))
    
    fill = cv2.contourArea(contours[largestIndex])/(math.sqrt(l1)*math.sqrt(l2))
    #% of rectangle is of path pixels
    

    fill_rect = 1
    #% of rectange's inside border of width w is of path pixels
    
    sides = [math.sqrt(l1), math.sqrt(l2)]
    ratio = max(sides)/min(sides) 
    #ratio of long side to short side

    noise = float(white_pixels/contour_area)
    #if noise is large, lots of orange particles not in "path"
    #print(noise)
    #print (blue)
    #print (ratio)
    
    if noise < 3:
        confidence = confidence + 0.7
    elif noise < 4:
        confidence = confidence + 0.3
    if ratio >10 and noise > 3:
        confidence = confidence + 0.5
    confidence = confidence + fill
    if confidence > 1:
        confidence = 1

#probability that path exists        
    return (r,theta), (x_center, y_center), confidence, dilation
		

}
	
	
	
	
	
	
	
	





