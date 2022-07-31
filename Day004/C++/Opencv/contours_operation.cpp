//
// Created by 刘 on 2022/7/26.
//#轮廓操作

#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

int main(){
    cv::Mat img=cv::imread("img/8.jpg");

    cv::Mat gray,bin;
    cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);
    cv::threshold(gray,bin,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);
    vector<vector<cv::Point>>contours;
    cv::findContours(bin,contours,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE);
//    cv::drawContours(img,contours,-1,cv::Scalar(0,0,255),1);
//    轮廓近似
//    cv::approxPolyDP(contours.at(0),contours.at(0),60, true);
//    cv::drawContours(img,contours,-1,cv::Scalar(0,0,255),1);
//    凸包修复
    vector<vector<cv::Point>>hull(contours.size());
    cv::convexHull(contours.at(0),hull.at(0));
    cout<<cv::isContourConvex(contours.at(0))<<endl<<cv::isContourConvex(hull.at(0));
    cv::drawContours(img,hull,0,cv::Scalar(0,0,255),2);
    cv::imshow("dst",img);
    cv::waitKey(0);
    cv::destroyAllWindows();

}

