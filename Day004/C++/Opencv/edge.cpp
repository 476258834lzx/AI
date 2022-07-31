//
// Created by 刘 on 2022/7/26.
//#边缘提取

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img=cv::imread("img/12.jpg",0);
    cv::Canny(img,img,50,150);
    cv::imshow("dst",img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}