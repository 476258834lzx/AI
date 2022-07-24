//
// Created by 刘 on 2022/7/22.
//#基本绘图

#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

int main(){
    cv::Mat img=cv::imread("img/4.jpg");
    cv::line(img,cv::Point(100,30),cv::Point(210,180),cv::Scalar(0,0,255),2);
    cv::circle(img,cv::Point (50,50),30,cv::Scalar(0,0,255),2);
    cv::ellipse(img,cv::Point(100,100),cv::Point(100,50),0,0,360,cv::Scalar(0,0,255),2);
    cv::rectangle(img,cv::Point(100,30),cv::Point(210,180),cv::Scalar(0,0,255),2);
    //绘制多边形
    vector<cv::Point> contor;
    contor.push_back(cv::Point(10,50));
    contor.push_back(cv::Point(20,71));
    contor.push_back(cv::Point(70,130));
    contor.push_back(cv::Point(139,153));
    cv::polylines(img,contor, true,cv::Scalar(0,0,255),2,cv::LINE_AA);
    //写字
    cv::putText(img,"mother fucker!",cv::Point(200,300),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(0,0,255),2);
    cv::imshow("img",img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}