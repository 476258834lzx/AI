//
// Created by 刘 on 2022/7/24.
//#形态学操作

//
// Created by 刘 on 2022/7/24.
//#仿射变换

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img=cv::imread("img/10.jpg");
    cv::Mat kernel=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    cv::Mat dst;
//    cv::dilate(img,dst,kernel);
//    cv::erode(img,dst,kernel);
//    cv::morphologyEx(img,dst,cv::MORPH_OPEN,kernel);
//    cv::morphologyEx(img,dst,cv::MORPH_CLOSE,kernel);
//    cv::morphologyEx(img,dst,cv::MORPH_TOPHAT,kernel);
//    cv::morphologyEx(img,dst,cv::MORPH_BLACKHAT,kernel);
    cv::morphologyEx(img,dst,cv::MORPH_GRADIENT,kernel);
    cv::imshow("img",img);
    cv::imshow("dst",dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

