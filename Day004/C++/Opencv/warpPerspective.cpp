//
// Created by 刘 on 2022/7/24.
//#透视变换

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img=cv::imread("img/10.jpg");
    cv::Point2f pts1[]={cv::Point2f (25,30),cv::Point2f (179,25),cv::Point2f (12,188),cv::Point2f (189,190)};
    cv::Point2f pts2[]={cv::Point2f (0,0),cv::Point2f (200,0),cv::Point2f (0,200),cv::Point2f (200,200)};
    cv::Mat M=cv::getPerspectiveTransform(pts1,pts2);
    cv::Mat dst;
    cv::warpPerspective(img,dst,M,img.size());
    cv::imshow("img",img);
    cv::imshow("dst",dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}