//
// Created by 刘 on 2022/7/20.
//

#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

int main(){
    cv::Mat img=cv::Mat(200,200,CV_8UC3,cv::Scalar(255,0,0));//形状，元素类型(无符号int8三通道)，颜色，scalar类别
    //操作每个通道
//    for (int i=0;i<img.rows;++i){
//        for (int j=0;j<img.cols;++j){
//            img.at<cv::Vec3b>(i,j)[0]=0;
//            img.at<cv::Vec3b>(i,j)[1]=0;
//            img.at<cv::Vec3b>(i,j)[2]=255;
//        }
//    }
    vector<cv::Mat> ms;
//    通道切割
    cv::split(img,ms);
    ms[0]=cv::Scalar(0);
    ms[1]=cv::Scalar(255);
    ms[2]=cv::Scalar(0);
//    合并通道
    cv::merge(ms,img);
    cv::imshow("img",img);
    cv::waitKey(0);
    cv::destroyAllWindows();
//    保存图片
    cv::imwrite("img/save.jpg",img);
}