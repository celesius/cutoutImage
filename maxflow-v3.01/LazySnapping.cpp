//
//  LazySnapping.cpp
//  opencv_test
//
//  Created by vk on 15/7/21.
//  Copyright (c) 2015年 clover. All rights reserved.
//

#include "LazySnapping.h"

using namespace std;

LasySnapping::LasySnapping():graph(NULL)
{
    //avgForeColor ={0,0,0};
    //avgBackColor ={0,0,0};
    avgForeColor[0] = 0;
    avgForeColor[1] = 0;
    avgForeColor[2] = 0;
    avgBackColor[0] = 0;
    avgBackColor[1] = 0;
    avgBackColor[2] = 0;
    
}

LasySnapping::~LasySnapping()
{
    if(graph){
        delete graph;
    }
};


void LasySnapping::setImage(IplImage* image){
    this->image = image;
    graph = new GraphType(image->width*image->height,image->width*image->height*2);
}

void LasySnapping::setForegroundPoints(std::vector<CvPoint> pts){
    forePts.clear();
    for(int i =0; i< pts.size(); i++){
        if(!isPtInVector(pts[i],forePts)){
            forePts.push_back(pts[i]);
        }
    }
    if(forePts.size() == 0){
        return;
    }
    int sum[3] = {0};
    for(int i =0; i < forePts.size(); i++){
        unsigned char* p = (unsigned char*)image->imageData + forePts[i].x * 3
        + forePts[i].y*image->widthStep;
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
    }
    cout<<sum[0]<<" " <<forePts.size()<<endl;
    avgForeColor[0] = sum[0]/forePts.size();
    avgForeColor[1] = sum[1]/forePts.size();
    avgForeColor[2] = sum[2]/forePts.size();
}
void LasySnapping::setBackgroundPoints(vector<CvPoint> pts)
{
    backPts.clear();
    for(int i =0; i< pts.size(); i++){
        if(!isPtInVector(pts[i],backPts)){
            backPts.push_back(pts[i]);
        }
    }
    if(backPts.size() == 0){
        return;
    }
    int sum[3] = {0};
    for(int i =0; i < backPts.size(); i++){
        unsigned char* p = (unsigned char*)image->imageData + backPts[i].x * 3 +
        backPts[i].y*image->widthStep;
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
    }
    avgBackColor[0] = sum[0]/backPts.size();
    avgBackColor[1] = sum[1]/backPts.size();
    avgBackColor[2] = sum[2]/backPts.size();
}


float LasySnapping::colorDistance(unsigned char* color1, unsigned char* color2)
{
    return sqrt((color1[0]-color2[0])*(color1[0]-color2[0])+
                (color1[1]-color2[1])*(color1[1]-color2[1])+
                (color1[2]-color2[2])*(color1[2]-color2[2]));
}

//float LasySnapping::minDistance(unsigned char* color, vector<CvPoint> points)
//{
//    float distance = -1;
//    for(int i =0 ; i < points.size(); i++){
//        unsigned char* p = (unsigned char*)image->imageData + points[i].y * image->widthStep +
//            points[i].x * image->nChannels;
//        float d = colorDistance(p,color);
//        if(distance < 0 ){
//            distance = d;
//        }else{
//            if(distance > d){
//                distance = d;
//            }
//        }
//    }
//}


bool LasySnapping::isPtInVector(CvPoint pt, vector<CvPoint> points)
{
    for(int i =0 ; i < points.size(); i++){
        if(pt.x == points[i].x && pt.y == points[i].y){
            return true;
        }
    }
    return false;
}
void LasySnapping::getE1(unsigned char* color,float* energy)
{
    // average distance
    float df = colorDistance(color,avgForeColor);
    float db = colorDistance(color,avgBackColor);
    // min distance from background points and forground points
    // float df = minDistance(color,forePts);
    // float db = minDistance(color,backPts);
    energy[0] = df/(db+df);
    energy[1] = db/(db+df);
}
float LasySnapping::getE2(unsigned char* color1,unsigned char* color2)
{
    const float EPSILON = 0.01;
    float lambda = 100;
    return lambda/(EPSILON+
                   (color1[0]-color2[0])*(color1[0]-color2[0])+
                   (color1[1]-color2[1])*(color1[1]-color2[1])+
                   (color1[2]-color2[2])*(color1[2]-color2[2]));
}

int LasySnapping::runMaxflow()
{
    const float INFINNITE_MAX = 1e10;
    int indexPt = 0;
    for(int h = 0; h < image->height; h ++){
        unsigned char* p = (unsigned char*)image->imageData + h *image->widthStep;
        for(int w = 0; w < image->width; w ++){
            // calculate energe E1
            float e1[2]={0};
            if(isPtInVector(cvPoint(w,h),forePts)){
                e1[0] =0;
                e1[1] = INFINNITE_MAX;
            }else if(isPtInVector(cvPoint(w,h),backPts)){
                e1[0] = INFINNITE_MAX;
                e1[1] = 0;
            }else {
                getE1(p,e1);
            }
            // add node
            graph->add_node();
            graph->add_tweights(indexPt, e1[0],e1[1]);
            // add edge, 4-connect
            if(h > 0 && w > 0){
                float e2 = getE2(p,p-3);
                graph->add_edge(indexPt,indexPt-1,e2,e2);
                e2 = getE2(p,p-image->widthStep);
                graph->add_edge(indexPt,indexPt-image->width,e2,e2);
            }
            
            p+= 3;
            indexPt ++;
        }
    }
    
    return graph->maxflow();
}
IplImage* LasySnapping::getImageMask()
{
    IplImage* gray = cvCreateImage(cvGetSize(image),8,1);
    int indexPt =0;
    for(int h =0; h < image->height; h++){
        unsigned char* p = (unsigned char*)gray->imageData + h*gray->widthStep;
        for(int w =0 ;w <image->width; w++){
            if (graph->what_segment(indexPt) == GraphType::SOURCE){
                *p = 0;
            }else{
                *p = 255;
            }
            p++;
            indexPt ++;
        }
    }
    return gray;
}
