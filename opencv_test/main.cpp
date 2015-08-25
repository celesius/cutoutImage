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
#include "CutoutImagePacking.h"

//using namespace std;

//cv::Mat classCutMat; //全局抠图结果。存储最终抠图数据，用于椭圆拟合
//CutoutImage *cutoutImage;
CutoutImagePacking *cutoutImagePacking;
int mousePointWidth = 0;
int selectSeedMat = 0;
std::vector<cv::Mat> seedMatVector;
int maxSelectSeedMat = 0;
cv::Mat seedStoreMat;
bool addMask = false;

//这个用于测试生长算法
//void on_mouse( int event, int x, int y, int flags, void* param)
//{
//    static int cnt = 0;
//    cnt++;
//    //cout<<"CV_EVENT_LBUTTONDOWN "<< cnt <<endl;
//    cv::Mat receiveMat((IplImage *)param,false); //接收传入的
//    /*
//     *生长算法需要的数据
//     */
//    cv::Mat static growDstMat = cv::Mat(receiveMat.rows,receiveMat.cols,CV_8UC1,cv::Scalar(0));
//    IplImage growSrcIpl = receiveMat;
//    IplImage growDstIpl = growDstMat;
//    
//    if( event == CV_EVENT_LBUTTONDOWN ){
//        char pixelCoordText[100];
//        int grayData = receiveMat.at<uchar>(y,x);
//        sprintf(pixelCoordText, " (%d,%d)%d",x,y,grayData);
//        /*
//         *开始生长算法
//         */
//        cutoutImage->regionGrowClover(receiveMat,growDstMat,x,y,3);
//        cv::Mat static receiveMatClone = receiveMat.clone();
//        cv::circle(receiveMatClone, cv::Point(x,y), 2, cv::Scalar(255));
//        cv::putText(receiveMatClone, pixelCoordText, cv::Point(x,y), CV_FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(100));
//        cv::imshow("orgGray", receiveMatClone);
//        //RegionGrow(&growSrcIpl, &growDstIpl, x, y,10, 1);
//        cv::imshow("mouseGrow", growDstMat);
//    }
//}



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

//这个用于测试局部算法
void on_mouse_cube( int event, int x, int y, int flags, void* param)
{
    //首先要记录滑动区域并统计计算窗口大小
    cv::Mat showMat = cv::Mat((IplImage *)param,true);
    cv::Mat showMatClone = showMat.clone(); //用于画点，画线显示用
    //cv::Mat showMergeColorImg = showMat.clone();
    cv::Mat shoMatMouse = showMat.clone();
    //static cv::Mat seedStoreMat = cv::Mat( showMat.rows, showMat.cols, CV_8UC1, cv::Scalar(0) );
    //static cv::Mat allShow ;//= showMat.clone();
    forePts.clear();
    std::vector<cv::Point> static mouseSlideRegionDiscrete;
    cv::circle(shoMatMouse, cv::Point(x,y), mousePointWidth/2, cv::Scalar(0));
    //cv::imshow("shoMatMouse", shoMatMouse);
    //std::vector<cv::Point> static mouseSlideRegion; //鼠标按下滑动区域
    if(event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)){  //按下左键并移动
        //if(event == (CV_EVENT_MOUSEMOVE && CV_EVENT_FLAG_LBUTTON)){
        mouseSlideRegionDiscrete.push_back(cv::Point(x,y));
        //7*7种子点
    }
    else if(event == CV_EVENT_LBUTTONDOWN)
    {
        mouseSlideRegionDiscrete.push_back(cv::Point(x,y));
    }
    else if(event == CV_EVENT_LBUTTONUP){
        if(addMask == true){
            cv::Mat drawMaskResult;
            cutoutImagePacking->drawMask(mouseSlideRegionDiscrete, mousePointWidth, drawMaskResult);
            cv::imshow("orgGray", drawMaskResult);
          /*
            if((int)seedMatVector.size() != 0){
                cv::Mat sendSeedStoreMat = seedMatVector[selectSeedMat].clone();  //这个一定要注意否则就把当前拿出的mat修改了
                cv::imshow("sendSeedStoreMatA", sendSeedStoreMat);
                cutoutImage->processImageAddMask( mouseSlideRegionDiscrete, sendSeedStoreMat, seedStoreMat, mousePointWidth, showMat);
            }
            else{
            
            }
            cv::Mat matWillSave = seedStoreMat.clone();
            //首先要删除selectSeedMat以后的内容，因为可能返回或者前进
            if((int)seedMatVector.size() != 0)
            {
                for(;;){
                    if( selectSeedMat + 1 == (int)seedMatVector.size() ){
                        break;
                    }
                    else{
                        seedMatVector.pop_back();
                    }
                }
            }
            seedMatVector.push_back(matWillSave);
            int vCnt = (int)seedMatVector.size();
            if( vCnt ==  maxSelectSeedMat + 1) //保证只有5个
            {
                seedMatVector.erase(seedMatVector.begin()); //删除最开始的mat
            }
            selectSeedMat = (int)seedMatVector.size() - 1;
            */
            mouseSlideRegionDiscrete.clear();
        }
        else{
            cv::Mat creatMaskResult;
            cutoutImagePacking->creatMask(mouseSlideRegionDiscrete, mousePointWidth, creatMaskResult);
            cv::imshow("orgGray", creatMaskResult);
            //drawLine();
            //std::vector<cv::Point> allSeedPoint;
            /*
            if((int)seedMatVector.size() != 0){ //为0是最开始
                cv::Mat sendSeedStoreMat = seedMatVector[selectSeedMat].clone();  //这个一定要注意否则就把当前拿出的mat修改了
                cv::imshow("sendSeedStoreMatC", sendSeedStoreMat);
                cutoutImage->processImageCreatMask(mouseSlideRegionDiscrete, showMat, sendSeedStoreMat,mousePointWidth,10);
                seedStoreMat = sendSeedStoreMat;
            }
            else{
                cutoutImage->processImageCreatMask(mouseSlideRegionDiscrete, showMat, seedStoreMat,mousePointWidth,10);
            }
            //这里要存储
            cv::Mat matWillSave = seedStoreMat.clone();
            //首先要删除selectSeedMat以后的内容，因为可能返回或者前进
            if((int)seedMatVector.size() != 0)
            {
                for(;;){
                    if( selectSeedMat + 1 == (int)seedMatVector.size() ){
                        break;
                    }
                    else{
                        seedMatVector.pop_back();
                    }
                }
            }
            seedMatVector.push_back(matWillSave);
            int vCnt = (int)seedMatVector.size();
            if( vCnt ==  maxSelectSeedMat + 1) //保证只有5个
            {
                seedMatVector.erase(seedMatVector.begin()); //删除最开始的mat
            }
            selectSeedMat = (int)seedMatVector.size() - 1; //只要有修改就将selectSeedMat放到修改位置
            
            //cv::Mat allShow = cutoutImage->getMergeResult();
            //cv::imshow("orgGray", allShow);
            //mouseSlideRegion.clear();
            */
            mouseSlideRegionDiscrete.clear();
            std::cout<<"CV_EVENT_LBUTTONUP" <<std::endl;
            std::cout<<" selectSeedMat =  " << selectSeedMat << std::endl;
        }
    }
    else if(event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_RBUTTON)){
        mouseSlideRegionDiscrete.push_back(cv::Point(x,y));  //擦除点数据填充
    }
    else if(event == CV_EVENT_RBUTTONDOWN){
        mouseSlideRegionDiscrete.push_back(cv::Point(x,y));
    }
    else if(event == CV_EVENT_RBUTTONUP){   //右键抬起  删除mark
        cv::Mat deletMaskResult;
        cutoutImagePacking->deleteMask(mouseSlideRegionDiscrete, mousePointWidth, deletMaskResult);
        cv::imshow("orgGray", deletMaskResult);
        /*
        if((int)seedMatVector.size() != 0){
            seedStoreMat = seedMatVector[selectSeedMat].clone();
        }
        else{
            seedStoreMat = cv::Mat(showMat.rows,showMat.cols,CV_8UC1,cv::Scalar(0));
        }
        
        cv::Mat emptyMat;
        cutoutImage->processImageDeleteMask(mouseSlideRegionDiscrete,seedStoreMat,showMat,emptyMat,mousePointWidth);
        
        if((int)seedMatVector.size() != 0)  //若已经生成过计算结果
        {
            cv::Mat matWillBeStore = seedStoreMat.clone();
            seedMatVector.push_back(matWillBeStore);
            int vCnt = (int)seedMatVector.size();
            if( vCnt == maxSelectSeedMat + 1 ) //保证只有设置的最大数量
            {
                seedMatVector.erase(seedMatVector.begin());
            }
            selectSeedMat = (int)seedMatVector.size() - 1;
        }
        //        cv::imshow("orgGray", allShow);
        //        cv::imshow("deleteMat", deleteMat);
        */
        mouseSlideRegionDiscrete.clear();
        std::cout<<"CV_EVENT_RBUTTONUP" <<std::endl;
        std::cout<<" selectSeedMat =  " << selectSeedMat << std::endl;
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

void initSeedMatVector(cv::Size matSize)
{
    seedMatVector.clear();
    cv::Mat initZeroMat = cv::Mat(matSize.height,matSize.width,CV_8UC1,cv::Scalar(0));
    seedMatVector.push_back(initZeroMat);
}

int main(int argc, char** argv)
{
    cv::Mat img = cv::imread("/Users/vk/Pictures/SkinColorImg/texture/2.jpg");
    // cv::imshow("org", img);
    cv::Mat dst = cv::Mat(img.rows,img.cols,CV_8UC1,cv::Scalar(0));
    cv::Mat grayImg;
    cv::cvtColor(img, grayImg, CV_BGR2GRAY);
    maxSelectSeedMat = 20;
    mousePointWidth = 10;
    cutoutImagePacking = new CutoutImagePacking;
    cutoutImagePacking->setColorImage(img, 20);
    //cutoutImage = new CutoutImage;  //测试取消
    //初始化seedMatVector;
    cv::Size aSize;
    aSize.width = img.cols;
    aSize.height = img.rows;
    initSeedMatVector(aSize);
    seedStoreMat = cv::Mat( img.rows, img.cols, CV_8UC1, cv::Scalar(0) );
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
    cv::Mat mainMat;
    while (1)
    {
        int key = cv::waitKey(2);
        if(key =='b') //回退
        {
            cv::Mat redoMat;
            cutoutImagePacking->redo(redoMat);
            cv::imshow("orgGray", redoMat);
            /*
            if((int)seedMatVector.size() != 0 && selectSeedMat != 0)  //
            {
                selectSeedMat --;
                cutoutImage->colorDispResultWithFullSeedMat(grayImg, seedMatVector[selectSeedMat]);
            }
            */
            std::cout<<"key BB selectSeedMat = " << selectSeedMat <<std::endl;
        }
        else if(key == 'r') //重做
        {
            cv::Mat resetMat;
            cutoutImagePacking->resetMask(resetMat);
            cv::imshow("orgGray", resetMat);
            /*
            selectSeedMat = 0;
            initSeedMatVector(aSize);
            seedStoreMat = cv::Mat( img.rows, img.cols, CV_8UC1, cv::Scalar(0) );
            cutoutImage->colorDispResultWithFullSeedMat(grayImg, seedMatVector[selectSeedMat]);
            */
        }
        else if(key == 'f') //前进
        {
            cv::Mat undoMat;
            cutoutImagePacking->undo(undoMat);
            cv::imshow("orgGray", undoMat);
            /*
            if( selectSeedMat !=  maxSelectSeedMat - 1 && selectSeedMat != seedMatVector.size() - 1 ){
                selectSeedMat ++;
                cutoutImage->colorDispResultWithFullSeedMat(grayImg, seedMatVector[selectSeedMat]);
            }
             */
            std::cout<<"key FF = seedMatVector.size()" << seedMatVector.size() <<std::endl;
            std::cout<<"key FF selectSeedMat = " << selectSeedMat <<std::endl;
        }
        else if(key == 'd') //完成抠图进入下一步，自动旋转旋转
        {
            //cutoutImage->rotateMat(cutoutImage->classCutMat, mainMat,img);
        }
        else if(key == '=')
        {
            mousePointWidth ++;
        }
        else if(key == '-')
        {
            mousePointWidth --;
        }
        else if(key == 'h'){
            
            for(int i=0;i<(int)seedMatVector.size();i++)
            {
                char mynum[100];
                sprintf(mynum, "show %d",i);
                cv::imshow(mynum, seedMatVector[i]);
            }
        }
        else if(key == 'p') //图像后处理，首先要处理得到的抠图blob,将锐利边缘的二值图转换为平滑边缘的，再利用图像融合将锐利边缘的图像与平滑边缘的图像融合，后续需要进行的一个工作
        {
            cv::Mat dstMat;
            dstMat = cutoutImagePacking->getFinalColorMergeImg();
            /*
            cv::Mat blobMat = seedMatVector[selectSeedMat];
            cv::Mat dstMat;
            cutoutImage->edgeBlur( img, blobMat, 21, dstMat );  //dstMat就是扣取结果，还要对结果进行椭圆拟合和旋转
            cutoutImage->rotateMat(cutoutImage->classCutMat, mainMat, dstMat);
            */
            cv::imshow( " cutResult ", dstMat);
        }
        else if(key == 'a') //直接添加mask不用算法
        {
            addMask = true;
            std::cout << "-----直接添加MASK-----" <<std::endl;
        }
        else if(key == 'c') //算法计算
        {
            addMask = false;
            std::cout << "-----算法计算MASK-----" <<std::endl;
        }
            
        else if(key == 'q') //退出
            break;
    }
    return 0;
}