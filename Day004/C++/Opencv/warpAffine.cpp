//
// Created by 刘 on 2022/7/24.
//#仿射变换

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img=cv::imread("img/10.jpg");
//    cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,50,0,1,50);
    cv::Mat M=cv::getRotationMatrix2D(cv::Point(img.cols/2,img.rows/2),45,0.7);
    cv::Mat dst;
    cv::warpAffine(img,dst,M,img.size());
    cv::imshow("img",img);
    cv::imshow("dst",dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

