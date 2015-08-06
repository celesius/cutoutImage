//
//  cvRoundRectangle.cpp
//  opencv_test
//
//  Created by vk on 15/8/6.
//  Copyright (c) 2015年 clover. All rights reserved.
//

#include "cvRoundRectangle.h"
#include <math.h>

//void cvRoundRectangle(IplImage *image, cv::Point lefttop, cv::Point rightbottom,int radius,
                      //CvScalar color,int thickness=1, int line_type=8, int shift=0)
void cvRoundRectangle(cv::Mat image, cv::Point lefttop, cv::Point rightbottom,int radius,cv::Scalar color,int thickness, int line_type, int shift)
{
    int temp;
    if(lefttop.x>rightbottom.x)
    {
        temp=lefttop.x;
        lefttop.x=rightbottom.x;
        rightbottom.x=temp;
    }
    if(lefttop.y>rightbottom.y)
    {
        temp=lefttop.y;
        lefttop.y=rightbottom.y;
        rightbottom.y=temp;
    }
    if(rightbottom.x-lefttop.x<2*radius || rightbottom.y-lefttop.y<2*radius)
    {
        radius=cv::min((rightbottom.x-lefttop.x)/2,(rightbottom.y-lefttop.y)/2);
    }
    cv::Point center;
    if(thickness>0)
    {
        center=cv::Point(rightbottom.x-radius,rightbottom.y-radius); //rb
        cv::ellipse(image,center,cvSize(radius,radius),0,0,90,color,1,line_type,shift);
        
        center=cv::Point(lefttop.x+radius,rightbottom.y-radius);
        cv::ellipse(image,center,cvSize(radius,radius),0,90,180,color,1,line_type,shift);
        center=cv::Point(lefttop.x+radius,lefttop.y+radius); //lt
        cv::ellipse(image,center,cvSize(radius,radius),0,180,270,color,1,line_type,shift);
        
        center=cv::Point(rightbottom.x-radius,lefttop.y+radius);
        cv::ellipse(image,center,cvSize(radius,radius),0,270,360,color,1,line_type,shift);
        
        cv::line(image,cv::Point(lefttop.x+radius,lefttop.y),cv::Point(rightbottom.x-radius,lefttop.y),color,1,line_type,shift);
        cv::line(image,cv::Point(lefttop.x+radius,rightbottom.y),cv::Point(rightbottom.x-radius,rightbottom.y),color,1,line_type,shift);
        cv::line(image,cv::Point(lefttop.x,lefttop.y+radius),cv::Point(lefttop.x,rightbottom.y-radius),color);
        cv::line(image,cv::Point(rightbottom.x,lefttop.y+radius),cv::Point(rightbottom.x,rightbottom.y-radius),color,1,line_type,shift);
    }
    else
    {
        center=cv::Point(rightbottom.x-radius,lefttop.y+radius);
        cv::ellipse(image,center,cvSize(radius,radius),0,0,90,color,-1,line_type,shift);
        center=cv::Point(lefttop.x+radius,lefttop.y+radius);
        cv::ellipse(image,center,cvSize(radius,radius),0,90,180,color,-1,line_type,shift);
        center=cv::Point(lefttop.x+radius,rightbottom.y-radius);
        cv::ellipse(image,center,cvSize(radius,radius),0,180,270,color,-1,line_type,shift);
        center=cv::Point(rightbottom.x-radius,rightbottom.y-radius);
        cv::ellipse(image,center,cvSize(radius,radius),0,270,360,color,-1,line_type,shift);
        cv::rectangle(image,cv::Point(lefttop.x+radius,lefttop.y),cv::Point(rightbottom.x-radius,lefttop.y+radius),color,-1,line_type,shift);
        cv::rectangle(image,cv::Point(lefttop.x+radius,rightbottom.y-radius),cv::Point(rightbottom.x-radius,rightbottom.y),color,-1,line_type,shift);
        cv::rectangle(image,cv::Point(lefttop.x,lefttop.y+radius),cv::Point(lefttop.x+radius,rightbottom.y-radius),color,-1,line_type,shift);
        cv::rectangle(image,cv::Point(rightbottom.x-radius,lefttop.y+radius),cv::Point(rightbottom.x,rightbottom.y-radius),color,-1,line_type,shift);
        cv::rectangle(image,cv::Point(lefttop.x+radius,lefttop.y+radius),cv::Point(rightbottom.x-radius,rightbottom.y-radius),color,-1,line_type,shift);
    }
}