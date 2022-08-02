//
// Created by 刘 on 2022/8/1.
//霍夫直线检测

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat img=cv::imread("img/16.jpg");
    cv::Mat gray_img,bin_img;
    cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img,bin_img,0,255,cv::THRESH_OTSU);
    vector<cv::Vec4f>plines;
    cv::HoughLinesP(bin_img,plines,1,CV_PI/180,130);
    for(size_t i=0;i<plines.size();i++){
        cv::Vec4f hline=plines[i];
        cv::line(img,cv::Point(hline[0],hline[1]),cv::Point(hline[2],hline[3]),cv::Scalar(0,0,255),2);
    }

    cv::imshow("img",img);
    cv::imshow("dst",bin_img);
    cv::waitKey(0);
    cv::destroyAllWindows();

}