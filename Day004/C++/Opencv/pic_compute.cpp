//
// Created by 刘 on 2022/7/24.
//#图像运算

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img1=cv::imread("img/10.jpg");
    cv::Mat img2=cv::imread("img/11.jpg");

    cv::Mat add;
    cv::addWeighted(img1,0.7,img2,0.3,0,add);
    cv::imshow("adddst",add);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
