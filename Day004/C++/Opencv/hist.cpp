//
// Created by 刘 on 2022/8/3.
//#直方图均衡化

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img=cv::imread("img/10.jpg",0);
    cv::Mat dst;

    cv::equalizeHist(img,dst);
    cv::imshow("img",img);
    cv::imshow("dst",dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

