//
//  LazySnapping.h
//  opencv_test
//
//  Created by vk on 15/7/21.
//  Copyright (c) 2015年 clover. All rights reserved.
//

#ifndef __opencv_test__LazySnapping__
#define __opencv_test__LazySnapping__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include "graph.h"
typedef Graph<float,float,float> GraphType;

class LasySnapping{
    
    public :
    unsigned char avgForeColor[3];
    unsigned char avgBackColor[3];
    LasySnapping();
//    :graph(NULL){
//        //avgForeColor ={0,0,0};
//        //avgBackColor ={0,0,0};
//        avgForeColor[0] = 0;
//        avgForeColor[1] = 0;
//        avgForeColor[2] = 0;
//        avgBackColor[0] = 0;
//        avgBackColor[1] = 0;
//        avgBackColor[2] = 0;
//        
//    }
    ~LasySnapping();
//    {
//        if(graph){
//            delete graph;
//        }
//    };
    private :
    std::vector<CvPoint> forePts;
    std::vector<CvPoint> backPts;
    IplImage* image;
    // average color of foreground points
    
    // average color of background points
    
    public :
    void setImage(IplImage* image);
    // include-pen locus
    void setForegroundPoints(std::vector<CvPoint> pts);

    // exclude-pen locus
    void setBackgroundPoints(std::vector<CvPoint> pts);

    // return maxflow of graph
    int runMaxflow();
    // get result, a grayscale mast image indicating forground by 255 and background by 0
    IplImage* getImageMask();
    private :
    float colorDistance(unsigned char* color1, unsigned char* color2);
    //float minDistance(unsigned char* color, vector<CvPoint> points);
    bool isPtInVector(CvPoint pt, std::vector<CvPoint> points);
    void getE1(unsigned char* color,float* energy);
    float getE2(unsigned char* color1,unsigned char* color2);
    
    GraphType *graph;
};

#endif /* defined(__opencv_test__LazySnapping__) */
