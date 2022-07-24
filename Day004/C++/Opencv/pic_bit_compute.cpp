//
// Created by 刘 on 2022/7/24.
//#图像位运算

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img1=cv::imread("img/10.jpg");
    cv::Mat img2=cv::imread("img/11.jpg");
    cv::Mat dst1,dst2,dst3,dst4;
    cv::bitwise_and(img1,img2,dst1);
    cv::bitwise_or(img1,img2,dst2);
    cv::bitwise_not(img1,dst3);
    cv::bitwise_xor(img1,img2,dst4);
    cv::imshow("ds1",dst1);
    cv::imshow("ds2",dst2);
    cv::imshow("ds3",dst3);
    cv::imshow("ds4",dst4);
    cv::waitKey(0);
    cv::destroyAllWindows();
}