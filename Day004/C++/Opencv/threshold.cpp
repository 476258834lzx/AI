//
// Created by 刘 on 2022/7/24.
//#阈值操作

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img=cv::imread("img/10.jpg",0);
    cv::Mat bin_img;
    cv::threshold(img,bin_img,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);
    cv::imshow("img",bin_img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
