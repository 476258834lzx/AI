#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    cv::Mat img=cv::imread("img/1.jpg");
    cv::imshow("img",img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
