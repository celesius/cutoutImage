//
//  cvRoundRectangle.h
//  opencv_test
//
//  Created by vk on 15/8/6.
//  Copyright (c) 2015年 clover. All rights reserved.
//

#ifndef __opencv_test__cvRoundRectangle__
#define __opencv_test__cvRoundRectangle__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>

void cvRoundRectangle(cv::Mat image, cv::Point lefttop, cv::Point rightbottom,int radius,
                      cv::Scalar color,int thickness=1, int line_type=8, int shift=0);


#endif /* defined(__opencv_test__cvRoundRectangle__) */
