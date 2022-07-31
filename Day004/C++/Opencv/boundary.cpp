//
// Created by 刘 on 2022/8/1.
//边界检测

#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

int main(){
    cv::Mat img=cv::imread("img/8.jpg");
    cv::Mat gray_img,bin_img;
    cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img,bin_img,0,255,cv::THRESH_OTSU);
    vector<vector<cv::Point>>contours;
    cv::findContours(bin_img,contours,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE);
//    边界矩形
//    cv::Rect rect=cv::boundingRect(contours.at(0));
//    cv::rectangle(img,cv::Point(rect.x,rect.y),cv::Point(rect.x+rect.width,rect.y+rect.height),cv::Scalar(0,0,255),2);
//    最小外接矩形
//    cv::minAreaRect(contours.at(0));//返回左上角点坐标和宽高，一定为正的矩形
//    cv::RotatedRect minRect=cv::minAreaRect(contours.at(0));//可以旋转的矩形坐标，返回四个点坐标
//    cv::Point2f vs[4];
//    minRect.points(vs);
//    std::vector<cv::Point>contour;
//    contour.push_back(vs[0]);
//    contour.push_back(vs[1]);
//    contour.push_back(vs[2]);
//    contour.push_back(vs[3]);
//    cv::polylines(img,contour, true,cv::Scalar(0,0,255),2);
//    最小外切圆
    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contours.at(0),center,radius);
    cv::circle(img,center,radius,cv::Scalar(0,0,255),2);
    cv::imshow("img",img);
    cv::waitKey(0);
    cv::destroyAllWindows();
};
