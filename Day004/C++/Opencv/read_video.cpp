//
// Created by 刘 on 2022/7/21.
//视频读取

#include <opencv2/opencv.hpp>
#include <conio.h>
using namespace std;

int main(){
    cv::VideoCapture cap;
    cap=cv::VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4");
    while (true){
        cv::Mat frame;
        cap>>frame;//按类型返回只能接收到图像返回值，接收不到布尔类型ret
        cv::imshow("video",frame);
        cv::waitKey(42);
        if(_kbhit()){
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
}