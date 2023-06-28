//
// Created by 刘 on 2022/8/1.
//#面积、周长、重心

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img=cv::imread("img/10.jpg");
    cv::Mat gray_img,bin_img;
    cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img,bin_img,0,255,cv::THRESH_OTSU);

    vector<vector<cv::Point>>contours;
    cv::findContours(bin_img,contours,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE);

    cv::Moments M=cv::moments(contours.at(0));
    int cx =M.m10/M.m00;
    int cy=M.m01/M.m00;
    cout<<cx<<","<<cy<<endl;
    double area =cv::contourArea(contours.at(0));
    cout<<area<<endl;
    double area_len=cv::arcLength(contours.at(0), true);
    cout<<area_len;

}