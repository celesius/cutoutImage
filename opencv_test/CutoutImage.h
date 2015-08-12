//
//  CutoutImage.h
//  opencv_test
//
//  Created by vk on 15/8/11.
//  Copyright (c) 2015年 Clover. All rights reserved.
//

#ifndef __opencv_test__CutoutImage__
#define __opencv_test__CutoutImage__

#include <stdio.h>
#include <opencv2/opencv.hpp>

class CutoutImage{
public:
    CutoutImage();
    ~CutoutImage();
    void processImageCreatMask(std::vector<cv::Point> mouseSlideRegionDiscrete , const cv::Mat srcMat, cv::Mat &dstMat , int lineWidth, int expandWidth );  //最后一个参数用于将生成的矩形向外扩展一些
    void processImageDeleteMask(std::vector<cv::Point> mouseSlideRegionDiscrete , cv::Mat &seedStoreMat,const cv::Mat srcMat, cv::Mat &dstMat , int lineWidth );//seedmat将被修改，因为删除了部分内容
    
    void colorDispResultWithFullSeedMat( const cv::Mat picMat, const cv::Mat seedMat ); //需要这个函数用于外部debug
    void rotateMat (const cv::Mat srcMat ,cv::Mat &dstMat,const cv::Mat colorMat);
    cv::Mat getMergeResult();
    
public:
    cv::Mat classCutMat;
    cv::Mat classMergeMat;
    
private:
    std::vector<cv::Point> mouseSlideRegion; //鼠标按下滑动区域
    //删除mark需要的函数
    void deleteMatCreat(std::vector<cv::Point> inPoint,cv::Size matSize, int lineWide ,cv::Mat &dstMat);
    void deleteMask(const cv::Mat deleteMat,cv::Mat &seedMat);
    //建立mark需要的函数
    bool regionGrowClover( const cv::Mat srcMat ,cv::Mat &dstMat, int initSeedx, int initSeedy, int threshold);
    bool regionGrowClover( const cv::Mat srcMat, const cv::Mat seedStoreMat ,cv::Mat &dstMat, int initSeedx, int initSeedy, int threshold);
    void drawLineAndMakePointSet(std::vector<cv::Point> inPoint,cv::Size matSize,int lineWide,std::vector<cv::Point> &pointSet);
    void mergeProcess(const cv::Mat srcImg,cv::Mat &dstImg);
    void filterImage(const cv::Mat imFrame,cv::Mat & outFrame);
    void rectRegionGrow( std::vector<cv::Point> seedVector, cv::Point rectMatOrg, const cv::Mat srcMat, const cv::Mat seedStoreMat , cv::Mat &dstMat);
    void storeSeed(cv::Mat &storeSeedMat,cv::Mat currentSeedMat,cv::Point cseedMatAnchorPoint);
    void colorDispResult(const cv::Mat picMat, cv::Mat cutPicBitMat, cv::Point cutPicAnchorPoint , cv::Mat &mergeColorMat);
    void line2PointSet(const cv::Mat lineMat,std::vector<cv::Point> &pointSet);
    void deleteBlackIsland(const cv::Mat srcBitMat ,cv::Mat &dstBitMat);
};

#endif /* defined(__opencv_test__CutoutImage__) */
