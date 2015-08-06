//
//  main.cpp
//  opencv_test
//
//  Created by vk on 15/7/31.
//  Copyright (c) 2015年 leisheng526. All rights reserved.
//

//#include "main.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "graph.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <opencv2/core/core_c.h>
#include "LazySnapping.h"
#include "cvRoundRectangle.h"

//using namespace std;

//struct myd//保存种子像素
//{
//    int x;
//    int y;
//}; //seedpoint;
//
//typedef myd seedpoint;
//
//void Grow(IplImage* src,IplImage* seed, int gray)//gray=255
//{
//  //  cv::stac
//   // cv::stack<seedpoint>seedS;
//    std::vector<cv::Point> seedS;
//    cv::Point point;
//    // 获取图像数据,保存种子区域
//    int height     = seed->height;
//    int width      = seed->width;
//    int step       = seed->widthStep;
//    int channels   = seed->nChannels;
//    uchar* seed_data    = (uchar *)seed->imageData;
//    uchar* src_data=(uchar *)src->imageData;
//    for(int i=0;i<height;i++)
//    {
//        for(int j=0;j<width;j++)
//        {
//            if(seed_data[i*step+j]==255)
//            {
//                point.x=i;
//                point.y=j;
//                seedS.push_back(point);
//            }
//        }
//    }
//    while(!seedS.empty())
//    {
//        cv::Point temppoint;
//        point=seedS.front();
//        seedS.back();
//        if((point.x>0)&&(point.x<(height-1))&&(point.y>0)&&(point.y<(width-1)))
//        {
//            if((seed_data[(point.x-1)*step+point.y]==0)&&(src_data[(point.x-1)*step+point.y]==gray))
//            {
//                seed_data[(point.x-1)*step+point.y]=255;
//                temppoint.x=point.x-1;
//                temppoint.y=point.y;
//                seedS.push_back(temppoint);
//            }
//            if((seed_data[point.x*step+point.y+1]==0)&&(src_data[point.x*step+point.y+1]==gray))
//            {
//                seed_data[point.x*step+point.y+1]=255;
//                temppoint.x=point.x;
//                temppoint.y=point.y+1;
//                seedS.push_back(temppoint);
//            }
//            if((seed_data[point.x*step+point.y-1]==0)&&(src_data[point.x*step+point.y-1]==gray))
//            {
//                seed_data[point.x*step+point.y-1]=255;
//                temppoint.x=point.x;
//                temppoint.y=point.y-1;
//                seedS.push_back(temppoint);
//            }
//            if((seed_data[(point.x+1)*step+point.y]==0)&&(src_data[(point.x+1)*step+point.y]==gray))
//            {
//                seed_data[(point.x+1)*step+point.y]=255;
//                temppoint.x=point.x+1;
//                temppoint.y=point.y;
//                seedS.push_back(temppoint);
//            }
//        }
//    }
//}
//鼠标指令，为了便于测试

bool regionGrowClover(  cv::Mat const srcMat, cv::Mat &dstMat, int initSeedx, int initSeedy, int threshold) //4邻域生长灰度图生长
{
    bool rState = false;
    std::vector<cv::Point> seedVector;
    cv::Point currentPoint = cv::Point(0,0);
    //dstMat = cv::Mat(srcMat.rows,srcMat.cols,CV_8UC1,cv::Scalar(0)); //灰度图，所以只有单通道8bit
    
    seedVector.push_back(cv::Point(initSeedx,initSeedy)); //种子阵列初始化
    dstMat.ptr<uchar>(initSeedy)[initSeedx] = 255; //设置当前点为种子点
    
    uchar currentPixelValue = srcMat.ptr<uchar>(initSeedy)[initSeedx]; //当前点
    
    while (!seedVector.empty()) {
        //cv::vector<cv::Point>::iterator iter;
        //iter = seedVector.begin();
        currentPoint = seedVector[0]; //取出第一个并进行计算
        int x = currentPoint.x;
        int y = currentPoint.y;
        seedVector.erase(seedVector.begin());
        //std::cout<<"seedVector.size() = "<<seedVector.size()<<std::endl;
        //取得原始图像素值与原始图四邻域像素值
        //uchar currentPixelValue = srcMat.ptr<uchar>(y)[x]; //当前点
        //判断左面
        if(x != 0) //不是最左边点,左边点有效
        {
            if(dstMat.ptr<uchar>(y)[x-1] == 0)  //取出左边点的seed标示,且这个点是0，这个点没有当过种子点
            {
                uchar currentLeftPValue = srcMat.ptr<uchar>(y)[x - 1];
                //cout<< "currentPixelValue = " << (int)currentPixelValue <<endl;
                //cout<< "currentLeftPValue = " << (int)currentLeftPValue <<endl;
                if(abs(currentPixelValue - currentLeftPValue) <= (uchar)threshold){
                    dstMat.ptr<uchar>(y)[x-1] = 255;
                    seedVector.push_back(cv::Point(x-1,y)); //符合计算条件，将左面点压人计算队列
                }
                //uchar currentDstLeftValue = dstMat.ptr<uchar>(y)[x - 1];
            }
        }
        
        if(y != 0) //不是最上面点，上面点有效
        {
            if(dstMat.ptr<uchar>(y-1)[x] == 0)
            {
                uchar currentUpPValue = srcMat.ptr<uchar>(y - 1)[x];
                if(abs(currentPixelValue - currentUpPValue) <= (uchar)threshold){
                    dstMat.ptr<uchar>(y-1)[x] = 255;
                    seedVector.push_back(cv::Point(x,y-1));
                }
            }
        }
        
        if(x != (srcMat.cols-1)) //不是最右面点，右面点有效
        {
            if(dstMat.ptr<uchar>(y)[x + 1] == 0)
            {
                uchar currentRightPValue = srcMat.ptr<uchar>(y)[x + 1];
                if(abs(currentPixelValue - currentRightPValue) <= (uchar)threshold){
                    dstMat.ptr<uchar>(y)[x + 1] = 255;
                    seedVector.push_back(cv::Point(x+1,y));
                }
            }
        }
        
        if( y != (srcMat.rows - 1)) //不是最下面的点
        {
            if(dstMat.ptr<uchar>(y+1)[x] == 0)
            {
                uchar currentDownPValue = srcMat.ptr<uchar>(y + 1)[x];
                if(abs(currentPixelValue - currentDownPValue) <= (uchar)threshold){
                    dstMat.ptr<uchar>(y+1)[x] = 255;
                    seedVector.push_back(cv::Point(x,y+1));
                }
                
            }
        }
        
    }
    
    return rState;
}

//这个用于测试生长算法
void on_mouse( int event, int x, int y, int flags, void* param)
{
    static int cnt = 0;
    cnt++;
    //cout<<"CV_EVENT_LBUTTONDOWN "<< cnt <<endl;
    cv::Mat receiveMat((IplImage *)param,false); //接收传入的
    /*
     *生长算法需要的数据
     */
    cv::Mat static growDstMat = cv::Mat(receiveMat.rows,receiveMat.cols,CV_8UC1,cv::Scalar(0));
    IplImage growSrcIpl = receiveMat;
    IplImage growDstIpl = growDstMat;
    
    if( event == CV_EVENT_LBUTTONDOWN ){
        char pixelCoordText[100];
        int grayData = receiveMat.at<uchar>(y,x);
        sprintf(pixelCoordText, " (%d,%d)%d",x,y,grayData);
        /*
         *开始生长算法
         */
        regionGrowClover(receiveMat,growDstMat,x,y,3);
        cv::Mat static receiveMatClone = receiveMat.clone();
        cv::circle(receiveMatClone, cv::Point(x,y), 2, cv::Scalar(255));
        cv::putText(receiveMatClone, pixelCoordText, cv::Point(x,y), CV_FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(100));
        cv::imshow("orgGray", receiveMatClone);
        //RegionGrow(&growSrcIpl, &growDstIpl, x, y,10, 1);
        cv::imshow("mouseGrow", growDstMat);
    }
}

void filterImage(const cv::Mat imFrame,cv::Mat & outFrame)
{
    
    /* Soften image */
    cv::Mat tmpMat;
    cv::GaussianBlur(imFrame, tmpMat, cv::Size(5,5), 0,0);
    //myImageShow("Gaussian", tmpMat, cv::Size(640,480));
    //cvSmooth(ctx->image, ctx->temp_image3, CV_GAUSSIAN, 11, 11, 0, 0);
    /* Remove some impulsive noise */
    cv::medianBlur(tmpMat, tmpMat,1);
    //    cvSmooth(ctx->temp_image3, ctx->temp_image3, CV_MEDIAN, 11, 11, 0, 0);
    //cv::cvtColor(tmpMat, tmpMat, CV_BGR2HSV);
    //    cvCvtColor(ctx->temp_image3, ctx->temp_image3, CV_BGR2HSV);
    //
    //    /*
    //     * Apply threshold on HSV values to detect skin color
    //     */
    
    //cv::inRange(tmpMat, cv::Scalar(0,55,90,255), cv::Scalar(28,175,230,255), outFrame);
    
    //    cvInRangeS(ctx->temp_image3,
    //               cvScalar(0, 55, 90, 255),
    //               cvScalar(28, 175, 230, 255),
    //               ctx->thr_image);
    //
    //    /* Apply morphological opening */
    //    cvMorphologyEx(ctx->thr_image, ctx->thr_image, NULL, ctx->kernel,
    //                   CV_MOP_OPEN, 1);
    //    cvSmooth(ctx->thr_image, ctx->thr_image, CV_GAUSSIAN, 3, 3, 0, 0);
    
    //  IplConvKernel *kernel = cvCreateStructuringElementEx(9, 9, 4, 4, CV_SHAPE_RECT,NULL);
    
    
    cv::Mat kernelMat = getStructuringElement(CV_SHAPE_RECT, cv::Size(3,3),cv::Point(2,2));
    cv::morphologyEx(tmpMat, outFrame, CV_MOP_OPEN,kernelMat);
    //cv::GaussianBlur(outFrame, outFrame, cv::Size(3,3), 0,0);
    
}

void rectRegionGrow( std::vector<cv::Point> seedVector, cv::Point rectMatOrg, const cv::Mat srcMat, cv::Mat &dstMat)
{
    int matRow = srcMat.rows;
    int matCol = srcMat.cols;
    
    dstMat = cv::Mat(matRow,matCol,CV_8UC1,cv::Scalar(0));
    
    for(int i = 0;i<seedVector.size();i++){
        int seedx = seedVector[i].x - rectMatOrg.x;
        int seedy = seedVector[i].y - rectMatOrg.y;
        regionGrowClover(srcMat, dstMat, seedx, seedy, 5);
    }
    
}

std::vector<CvPoint> forePts;
std::vector<CvPoint> backPts;
const int SCALE = 4;
//IplImage* image = NULL;
char winName[] = "lazySnapping";
IplImage* imageDraw = NULL;

cv::Mat imageProcessingLS( const cv::Mat srcImage ){
    //cv::imshow("srcImage", srcImage);
    IplImage simage = srcImage;
    IplImage *image = &simage;
    
    //imageDraw = cvCloneImage(image);
    
    
    LasySnapping ls;
    
    IplImage* imageLS = cvCreateImage(cvSize(image->width/SCALE,image->height/SCALE),
                                      8,3);
    cvResize(image,imageLS);
    ls.setImage(imageLS);
    ls.setBackgroundPoints(backPts);
    ls.setForegroundPoints(forePts);
    double t = (double)cvGetTickCount();
    ls.runMaxflow();
    t = (double)cvGetTickCount() - t;
    printf( "run time = %gms\n", t/(cvGetTickFrequency()*1000) );
    IplImage* mask = ls.getImageMask();
    IplImage* gray = cvCreateImage(cvGetSize(image),8,1);
    cvResize(mask,gray);
    //cvShowImage("maskLS", mask);
    //cvShowImage("grayLS", gray);
    
    cv::Mat lsMat = cv::Mat(gray,true);
    // edge
    //    cvCanny(gray,gray,50,150,3);
    //
    //    IplImage* showImg = cvCloneImage(imageDraw);
    //    for(int h =0; h < image->height; h ++){
    //        unsigned char* pgray = (unsigned char*)gray->imageData + gray->widthStep*h;
    //        unsigned char* pimage = (unsigned char*)showImg->imageData + showImg->widthStep*h;
    //        for(int width  =0; width < image->width; width++){
    //            if(*pgray++ != 0 ){
    //                pimage[0] = 0;
    //                pimage[1] = 255;
    //                pimage[2] = 0;
    //            }
    //            pimage+=3;
    //        }
    //    }
    // cvSaveImage("t.bmp",showImg);
    // cvShowImage(winName,showImg);
    
    
    //cvReleaseImage(&image);
    cvReleaseImage(&imageLS);
    cvReleaseImage(&mask);
    // cvReleaseImage(&image);
    // cvReleaseImage(&showImg);
    cvReleaseImage(&gray);
    return lsMat;
    
}

void rectLazySnapping( std::vector<cv::Point> seedVector, cv::Point rectMatOrg, const cv::Mat srcMat, cv::Mat &dstMat )
{
    int matRow = srcMat.rows;
    int matCol = srcMat.cols;
    
    //dstMat = cv::Mat(matRow,matCol,CV_8UC1,cv::Scalar(0));
    forePts.clear();
    for(int i = 0;i<seedVector.size();i++){
        int seedx = seedVector[i].x - rectMatOrg.x;
        int seedy = seedVector[i].y - rectMatOrg.y;
        
        CvPoint setP;
        setP.x = seedx;
        setP.y = seedy;
        forePts.push_back(setP);
        //regionGrowClover(srcMat, dstMat, seedx, seedy, 1);
    }
    
    dstMat = imageProcessingLS(srcMat);
    
}

void cannyAndShow(const cv::Mat matWillShow)
{
    cv::Mat cannyDst;
    cv::Canny(matWillShow, cannyDst, 100, 100);
    cv::imshow("cannyDst", cannyDst);
}

void deleteBlackIsland(const cv::Mat srcBitMat ,cv::Mat &dstBitMat)
{
    cv::Mat a_mat = srcBitMat.clone();
    std::vector<std::vector<cv::Point>> contours;
    dstBitMat = cv::Mat(a_mat.rows, a_mat.cols, CV_8UC1,cv::Scalar(0));
    cv::findContours(a_mat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    cv::drawContours(dstBitMat, contours, -1, cv::Scalar(255),CV_FILLED);
}

void mergeProcess(const cv::Mat srcImg,cv::Mat &dstImg)
{
    cv::Mat a_mat = srcImg.clone();
    cv::Mat showContours = cv::Mat(a_mat.rows,a_mat.cols,CV_8UC1,cv::Scalar(0));
    //cv::Mat showContours = cv::Mat(srcImg.rows,srcImg.cols,CV_8UC4,cv::Scalar(0,0,0,0));
    // cv::Mat
    
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> hiararchy;
    
    cv::findContours(a_mat, contours, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);
    cv::drawContours(showContours, contours, -1, cv::Scalar(255),CV_FILLED);
    
    //cv::imshow("contours", showContours);
    
    cv::Mat closeMat;
    
    cv::morphologyEx(showContours, closeMat, cv::MORPH_CLOSE, cv::Mat(11,11,CV_8U),cv::Point(-1,-1),1);
    cv::imshow("closeMat", closeMat);
    
    //cv::medianBlur(closeMat, closeMat,5);
    
    //因为膨胀和腐蚀带来了一些"孤岛"(断裂的色块)，所以要再做一次背景块去除
    cv::Mat dbCloseMat;
    deleteBlackIsland(closeMat,dbCloseMat);
    cv::medianBlur(dbCloseMat, dbCloseMat,5); //后作中值滤波
    
    cv::dilate(dbCloseMat, dbCloseMat, cv::Mat());
    //    cv::imshow("dbCloseMat", dbCloseMat);
    //    cv::imshow("medianBlurCloseMat", closeMat);
    dstImg = dbCloseMat;
}

void colorDispResult(const cv::Mat picMat, cv::Mat cutPicBitMat, cv::Point cutPicAnchorPoint , cv::Mat &mergeColorMat)
{
    cv::Mat showMat = picMat.clone();
    
    if(showMat.channels() == 3)
        cv::cvtColor(showMat, showMat, CV_BGR2BGRA);
    else if(showMat.channels() == 1)
        cv::cvtColor(showMat, showMat, CV_GRAY2BGRA);
    
    cv::Mat colorCutPic = cutPicBitMat.clone();
    cv::cvtColor(colorCutPic, colorCutPic, CV_GRAY2BGRA);
    
    int colorCutPicRows = colorCutPic.rows;
    int colorCutPicCols = colorCutPic.cols*colorCutPic.channels();
    
    cv::imshow("cutPicBitMat", cutPicBitMat);
    
    for(int y= 0;y<colorCutPicRows;y++ ){
        uchar *cutMatRowsData = colorCutPic.ptr<uchar>(y);
        uchar *showMatRowsData = showMat.ptr<uchar>(cutPicAnchorPoint.y+y);
        for(int x = 0; x<colorCutPicCols; x = x+4){
            uchar oneChannelData = cutMatRowsData[x];
            if(oneChannelData == 255) //b
            {
                cutMatRowsData[x + 1] = 0; //g
                cutMatRowsData[x + 2] = 0; //r
                cutMatRowsData[x + 3] = 100; //a
                showMatRowsData[cutPicAnchorPoint.x*4 + x + 0] = 255;
                showMatRowsData[cutPicAnchorPoint.x*4 + x + 1] = showMatRowsData[cutPicAnchorPoint.x*4 + x + 1]/2;
                showMatRowsData[cutPicAnchorPoint.x*4 + x + 2] = showMatRowsData[cutPicAnchorPoint.x*4 + x + 2]/2;
                showMatRowsData[cutPicAnchorPoint.x*4 + x + 3] = 100;
                //x = x + 3;
            }
        }
    }
    cv::imshow("colorCutPicRows", colorCutPic);
    cv::imshow("showMatColor",showMat);
}

void storeSeed(cv::Mat &storeSeedMat,cv::Mat currentSeedMat,cv::Point cseedMatAnchorPoint)
{
    int cSeedMatRows = currentSeedMat.rows;
    int cSeedMatCols = currentSeedMat.cols;
    
    for(int y = 0;y<cSeedMatRows;y++){
        uchar *csMatRowData = currentSeedMat.ptr<uchar>(y);
        uchar *ssMatRowData = storeSeedMat.ptr<uchar>(y + cseedMatAnchorPoint.y);
        for (int x = 0; x<cSeedMatCols; x++) {
            ssMatRowData[x+cseedMatAnchorPoint.x] = csMatRowData[x];
        }
    }
}

//这个用于测试局部算法
void on_mouse_cube( int event, int x, int y, int flags, void* param)
{
    //首先要记录滑动区域并统计计算窗口大小
    
    cv::Mat showMat = cv::Mat((IplImage *)param,true);
    cv::Mat showMatClone = showMat.clone(); //用于画点，画线显示用
    cv::Mat showMergeColorImg = showMat.clone();
    
    static cv::Mat seedStoreMat = cv::Mat(showMat.rows,showMat.cols,CV_8UC1,cv::Scalar(0));
    
    forePts.clear();
    
    std::vector<cv::Point> static mouseSlideRegion; //鼠标按下滑动区域
    
    if(event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)){  //按下左键并移动
        mouseSlideRegion.push_back(cv::Point(x,y));
        
        mouseSlideRegion.push_back(cv::Point(x - 3,y));
        mouseSlideRegion.push_back(cv::Point(x + 3,y));
        mouseSlideRegion.push_back(cv::Point(x - 6,y));
        mouseSlideRegion.push_back(cv::Point(x + 6,y));
        mouseSlideRegion.push_back(cv::Point(x,y - 3));
        mouseSlideRegion.push_back(cv::Point(x,y + 3));
        mouseSlideRegion.push_back(cv::Point(x,y - 6));
        mouseSlideRegion.push_back(cv::Point(x,y + 6));
        //7*7种子点
        
        //这里不安全
    }
    else if(event == CV_EVENT_LBUTTONUP){
        int lx = showMat.cols,rx = 0,ty = showMat.rows,by = 0;
        for(int i = 0;i<(int)mouseSlideRegion.size();i++)
        {
            std::cout<<"point  = "<< mouseSlideRegion[i]<<std::endl;
            cv::circle(showMatClone, mouseSlideRegion[i], 3, cv::Scalar(255)); //绘制现实点
            //最左面点x，最右面点x，最上面点y,最下面点y
            if(mouseSlideRegion[i].x < lx)
            {
                lx = mouseSlideRegion[i].x;
            }
            if(mouseSlideRegion[i].x > rx)
            {
                rx = mouseSlideRegion[i].x;
            }
            if(mouseSlideRegion[i].y <ty )
            {
                ty = mouseSlideRegion[i].y;
            }
            if(mouseSlideRegion[i].y > by){
                by = mouseSlideRegion[i].y;
            }
            //CvPoint forePtsCvPoint =
        }
        std::cout<<" lx " << lx << " rx " << rx << " ty " << ty << " by " << by <<std::endl;
        
        cv::Point ltP = cv::Point(lx,ty);
        cv::Point rtP = cv::Point(rx,ty);
        cv::Point lbP = cv::Point(lx,by);
        cv::Point rbP = cv::Point(rx,by);
        //要截取的图形
        int rectMatRow = by - ty + 1;
        int rectMatCol = rx - lx + 1;
        cv::Mat recMat = cv::Mat (rectMatRow,rectMatCol,CV_8UC1,cv::Scalar(0));
        cv::Mat recRoundMat = cv::Mat (rectMatRow,rectMatCol,CV_8UC1,cv::Scalar(0)); //圆角矩形
        cvRoundRectangle(recRoundMat, cv::Point(0,0), cv::Point(rectMatCol-1,rectMatRow-1),20,cv::Scalar(255),1, 8, 0); //圆角矩形
        cv::imshow("recRoundMat", recRoundMat);
        
        cv::rectangle(showMatClone, ltP, rbP, cv::Scalar(255),1); //画图形
        
        for(int y = 0;y<rectMatRow;y++){
            uchar *rectMatLineData = recMat.ptr<uchar>(y);
            uchar *orgMatLineData = showMat.ptr<uchar>(ty+y);
            for(int x = 0; x < rectMatCol; x++){
                rectMatLineData[x] = orgMatLineData[lx+x];
            }
        }
        
        cv::Mat bitMat;
        int blockSize = 25;
        int constValue = 10;
        cv::adaptiveThreshold(recMat, bitMat, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
        
        cv::Mat filterImg;
        filterImage(recMat,filterImg);
        cv::Mat nextImg = filterImg.clone();
        cv::Mat regionGrowMat;
        rectRegionGrow( mouseSlideRegion, ltP, filterImg, regionGrowMat);
        cv::imshow("regionGrowMat", regionGrowMat);
        cv::Mat mergeMat;
        mergeProcess(regionGrowMat,mergeMat);
        cv::imshow("mergeMat", mergeMat);
        storeSeed(seedStoreMat,mergeMat,ltP);
        cv::imshow("seedStoreMat", seedStoreMat);
        
        //rectLazySnapping(mouseSlideRegion, ltP, nextImg, regionGrowMat);
        //cannyAndShow(filterImg);
        //cv::imshow("recMat", recMat);
        //cv::imshow("bitMat", bitMat);
        cv::imshow("showMat", showMatClone);
        cv::Mat colorMergeMat;
        colorDispResult(showMergeColorImg,mergeMat,ltP,colorMergeMat);
        
        
        //cv::imshow("filterImg", filterImg);
        //cv::adaptiveThreshold(filterImg, bitMat, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
        //cv::imshow("bitMatFilter", bitMat);
        
        
        mouseSlideRegion.clear();
        std::cout<<"CV_EVENT_LBUTTONUP" <<std::endl;
    }
    
    
}

void on_mouse_cube_color( int event, int x, int y, int flags, void* param)
{
    cv::Mat showMat = cv::Mat((IplImage *)param,true);
    cv::Mat showMatClone = showMat.clone(); //用于画点，画线显示用
    forePts.clear();
    
    std::vector<cv::Point> static mouseSlideRegion; //鼠标按下滑动区域
    
    if(event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)){  //按下左键并移动
        mouseSlideRegion.push_back(cv::Point(x,y));
    }
    else if(event == CV_EVENT_LBUTTONUP){
        int lx = showMat.cols,rx = 0,ty = showMat.rows,by = 0;
        for(int i = 0;i<(int)mouseSlideRegion.size();i++)
        {
            std::cout<<"point  = "<< mouseSlideRegion[i]<<std::endl;
            cv::circle(showMatClone, mouseSlideRegion[i], 3, cv::Scalar(255)); //绘制现实点
            
            //最左面点x，最右面点x，最上面点y,最下面点y
            
            if(mouseSlideRegion[i].x < lx)
            {
                lx = mouseSlideRegion[i].x;
            }
            
            if(mouseSlideRegion[i].x > rx)
            {
                rx = mouseSlideRegion[i].x;
            }
            
            if(mouseSlideRegion[i].y <ty )
            {
                ty = mouseSlideRegion[i].y;
            }
            if(mouseSlideRegion[i].y > by){
                by = mouseSlideRegion[i].y;
            }
            //CvPoint forePtsCvPoint =
            
            
        }
        std::cout<<" lx " << lx << " rx " << rx << " ty " << ty << " by " << by <<std::endl;
        
        cv::Point ltP = cv::Point(lx,ty);
        cv::Point rtP = cv::Point(rx,ty);
        cv::Point lbP = cv::Point(lx,by);
        cv::Point rbP = cv::Point(rx,by);
        //要截取的图形
        int rectMatRow = by - ty + 1;
        int rectMatCol = (rx - lx + 1) * showMat.channels();
        cv::Mat recMat = cv::Mat (rectMatRow,rectMatCol,CV_8UC3,cv::Scalar(0,0,0));
        cv::rectangle(showMatClone, ltP, rbP, cv::Scalar(255),1); //画图形
        
        for(int y = 0;y<rectMatRow;y++){
            uchar *rectMatLineData = recMat.ptr<uchar>(y);
            uchar *orgMatLineData = showMat.ptr<uchar>(ty+y);
            for(int x = 0; x < rectMatCol; x++){
                rectMatLineData[x] = orgMatLineData[lx*showMat.channels()+x];
            }
        }
        cv::Mat regionLazySnappingMat;
        //rectRegionGrow( mouseSlideRegion, ltP, filterImg, regionGrowMat);
        rectLazySnapping(mouseSlideRegion, ltP, recMat, regionLazySnappingMat);
        
        cv::imshow("recMat", recMat);
        cv::imshow("showMat", showMatClone);
        cv::imshow("regionLazySnappingMat", regionLazySnappingMat);
        mouseSlideRegion.clear();
    }
}

void on_mouse_roundRectangle(int event, int x, int y, int flags, void* param) //测试圆角
{
    cv::Mat showMat = cv::Mat((IplImage *)param,true);
    std::vector<cv::Point> static mouseSlideRegion;
    
    if(event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)){
        mouseSlideRegion.push_back(cv::Point(x,y));
    }
    else if(event == CV_EVENT_LBUTTONUP){
        int lx = showMat.cols,rx = 0,ty = showMat.rows,by = 0;
        for(int i = 0;i<(int)mouseSlideRegion.size();i++)
        {
            std::cout<<"point  = "<< mouseSlideRegion[i]<<std::endl;
            cv::circle(showMat, mouseSlideRegion[i], 3, cv::Scalar(255)); //绘制现实点
            
            //最左面点x，最右面点x，最上面点y,最下面点y
            
            if(mouseSlideRegion[i].x < lx)
            {
                lx = mouseSlideRegion[i].x;
            }
            
            if(mouseSlideRegion[i].x > rx)
            {
                rx = mouseSlideRegion[i].x;
            }
            
            if(mouseSlideRegion[i].y <ty )
            {
                ty = mouseSlideRegion[i].y;
            }
            if(mouseSlideRegion[i].y > by){
                by = mouseSlideRegion[i].y;
            }
            //CvPoint forePtsCvPoint =
        }
        cv::Point ltP = cv::Point(lx,ty);
        cv::Point rtP = cv::Point(rx,ty);
        cv::Point lbP = cv::Point(lx,by);
        cv::Point rbP = cv::Point(rx,by);
        
        cvRoundRectangle(showMat, ltP, rbP,20,cv::Scalar(255),1, 8, 0);
        cv::imshow("showMat", showMat);
        mouseSlideRegion.clear();
    }
    
}

int main(int argc, char** argv){
    
    cv::Mat img = cv::imread("/Users/vk/Pictures/SkinColorImg/texture/2.jpg");
    // cv::imshow("org", img);
    cv::Mat dst = cv::Mat(img.rows,img.cols,CV_8UC1,cv::Scalar(0));
    cv::Mat grayImg;
    cv::cvtColor(img, grayImg, CV_BGR2GRAY);
    IplImage iplGray = grayImg;
    IplImage iplDst = dst;
    //RegionGrow(&iplGray,&iplDst,500,500,7,1);
    
    //dst = iplDst;
    // cv::imshow("orgGray", grayImg);
    // cv::imshow("dst", dst);
    //    IplImage *psend2OnMouse;
    //    IplImage send2OnMouse = grayImg;
    // psend2OnMouse = &send2OnMouse;
    
    
    /*
     *测试生长算法
     */
    //cv::Mat growSrc = img;//grayImg;
    cv::Mat growSrc = grayImg;
    //cv::cvtColor(img, grayImg, CV_BGR2GRAY);
    IplImage *psend2OnMouse;
    IplImage send2OnMouse = growSrc;
    psend2OnMouse = &send2OnMouse;
    //send2OnMouse= IplImage(grayImg);
    cv::imshow("orgGray", growSrc);
    
    //cannyAndShow(grayImg);
    
    
    //cvSetMouseCallback( "orgGray", on_mouse_cube_color, (void *)psend2OnMouse);
    cvSetMouseCallback( "orgGray", on_mouse_cube, (void *)psend2OnMouse);
    //cvSetMouseCallback( "orgGray", on_mouse_roundRectangle, (void *)psend2OnMouse);
    
    
    
    int key = cv::waitKey(0);
    return 0;
}


