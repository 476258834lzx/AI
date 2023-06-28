//
// Created by 刘 on 2022/7/26.
//#滤波

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img=cv::imread("img/12.jpg");
    cv::Mat dst;
//    自定义滤波
//    cv::Mat M=(cv::Mat_<double>(3,3)<<1,1,0,1,0,-1,0,-1,-1);
//    cv::filter2D(img,dst,-1,M);
//    低通滤波
//    cv::blur(img,dst,cv::Size(3,3));
//    cv::GaussianBlur(img,dst,cv::Size(3,3),1,1);
//    cv::medianBlur(img,dst,3);
//    cv::bilateralFilter(img,dst,9,75,75);
//    高通滤波
//    cv::Laplacian(img,dst,-1);
//    cv::Sobel(img,dst,-1,1,0 );
//    cv::Scharr(img,dst,-1,1,0);

    cv::imshow("dst",dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
