#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>

/*
#include "common/defs.hpp"
#include "common/config.hpp"
#include "vision/config.hpp"
#include "vision/vision.hpp"
#include "vision/buoys_vision.hpp"
#include "image/image.hpp"
*/

#define PI 3.1415

void vision_func(cv::Mat image)
{
    // Initial needed variables
    double confidence;
    double conf = 0;
    double object_probability = 0;

    // Set initial blue
    int blue = 0;

    // Set threshold scalars
    cv::Scalar lower(0,100,200);
    cv::Scalar upper(185, 230, 255);

    // Filter out until unneeded
    cv::Mat mask;
    cv::inRange(image, lower, upper, mask);
    while (cv::countNonZero(mask) < 700 && blue < 245)
    {
        cv::Scalar lower_high(0, 100, 200);
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

    // Finds pixels that are more orange than blue
    // If less than 700 pixels are found, then the requirements are loosened
    int n = 1;
    if (blue > 210)
    {
        n = (int) (object_probability/15);
    }

    // Dialates pixels to make finding large contours easier
    cv::Mat kernel(2, 1, CV_8UC1, cv::Scalar(1));
    cv::Mat dilation;
    cv::dilate(mask, dilation, kernel, cv::Point(-1, -1), n);

    // Threshold image
    cv::Mat ret, thresh;
    cv::threshold(dilation, thresh, 127, 255, cv::THRESH_BINARY);
    
    // Detect contours that hopefully represent path
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(thresh, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    
    // Find contour with largest area
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

    // If no contour with less than minimum area is found, ignore
    if (largestIndex < 0)
    {
        return;
    }

    // Convert to bgr
    int white_pixels = cv::countNonZero(dilation);
    cv::cvtColor(dilation, dilation, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point> cnt = contours[largestIndex];

    cv::RotatedRect rect = cv::minAreaRect(cnt);
    std::vector<cv::Point> box;
    cv::boxPoints(rect, box);
    cv::drawContours(dilation,box,0,cv::Scalar(0,255,0),2);

    // Draws box around largest contour
    double l1 = std::pow((box[0].x - box[1].x), 2) + std::pow((box[0].y - box[1].y), 2);
    double l2 = std::pow((box[1].x - box[2].x), 2) + std::pow((box[1].y - box[2].y), 2);
    int m = -1;

    // Inverted y values because down is +y
    double theta, bearing;
    if (l1 > l2) 
    {
        if (box[0].x - box[1].x == 0)
        {
            m = -1;
        }
        else
        {
            m = -1 * (box[0].y - box[1].y) / (box[0].x - box[1].x);
        }
    }
    else
    {
        if (box[1].x - box[2].x == 0)
        {
            m = -1; 
        }
        else
        {
            m = -1 * (box[1].y - box[2].y) / (box[1].x - box[2].x);
        }
    }

    if (m != -1)
    {
        theta = std::atan(m) * 180/PI;
    }
    else
    {
        theta = 90;
    }

    if (theta < 0)
    {
        theta += 180;
        bearing = 90 - theta;
        std::cout << bearing << std::endl;
    }

    float r;
    if (l1 > l2) 
    {
        r = std::sqrt(l1);
    }
    else
    {
        r = std::sqrt(l2);
    }

    // Finds center of path
    int x_center = (box[0].x + box[1].x + box[2].x + box[3].x) / 4;
    int y_center = (box[0].y + box[1].y + box[2].y + box[3].y) / 4;
    cv::circle(dilation, cv::Point(x_center, y_center), 2, cv::Scalar(255, 125, 125), 6, 8, 0);

    // Draws line in direction of path
    int length = 100;
    int x2 = (int)(x_center + length * std::cos(theta * PI / 180.0));
    int y2 = (int)(y_center - length * std::sin(theta * PI / 180.0));
    cv::line(dilation, cv::Point(x_center, y_center), cv::Point(x2, y2), cv::Scalar(100, 100, 255), 4, 8, 0);

    // Get contour area, fill, and scale
    double contour_area = cv::contourArea(contours[largestIndex], false);
    double fill = cv::contourArea(contours[largestIndex], false) / (std::sqrt(l1)*std::sqrt(l2));
    double scale;
    if (std::sqrt(1) > std::sqrt(l2))
    {
        scale = std::sqrt(1) / std::sqrt(2);
    }
    else 
    {
        scale = std::sqrt(2) / std::sqrt(1);
    }

    // Get noise
    double noise = (double)(white_pixels/contour_area);
    if (noise < 3) 
    {
        confidence = confidence + 0.7;
    }
    else if (noise < 4)
    {
        confidence = confidence + 0.3;
    }
    if (scale > 10 && noise > 3) 
    {
        confidence = confidence + 0.5;
        confidence = confidence + fill;
    }
    if (confidence > 1) 
    {
        confidence = 1;
    }
}


int main(int argc, char** argv)
{
    // Get the path
}

