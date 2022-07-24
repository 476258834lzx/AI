//
// Created by 刘 on 2022/7/24.
//#矩阵运算

#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat x=(cv::Mat_<uchar>(2,1)<<250,34);
    cv::Mat y=(cv::Mat_<uchar>(2,1)<<10,100);
    cv::Mat add,sub;
    cv::add(x,y,add);
    cv::subtract(x,y,sub);
    cout<<add<<endl;
    cout<<sub<<endl;

}
