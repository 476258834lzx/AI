//
// Created by 刘 on 2022/7/22.
//色彩空间转换

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img=cv::imread("img/3.jpg");
    cv::Mat dst;
    cv::cvtColor(img,dst,cv::COLOR_BGR2GRAY);
    cv::convertScaleAbs(img,dst,6,0);
    cv::imshow("img",dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
