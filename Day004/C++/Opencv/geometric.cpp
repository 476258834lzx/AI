//
// Created by 刘 on 2022/7/24.
//#几何变换

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img=cv::imread("img/6.jpg");
    cv::Mat dst;
//    cv::resize(img,dst,cv::Size(300,300));
//    cv::transpose(img,dst);
//    cv::flip(img,dst,0);
    cv::rotate(img,dst,0);
    cv::imshow("img",img);
    cv::imshow("dst",dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
