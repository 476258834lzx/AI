#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    try {
        cv::Mat image=cv::imread("test_img/35.jpg",0);
        torch::Tensor tensor_img=torch::from_blob(image.data,{1,image.rows*image.cols},torch::kByte).toType(torch::kFloat);
        tensor_img/=255.;
        std::cout<<tensor_img.sizes()<<std::endl;
        //加载模型
        auto model=torch::jit::load("minist.pt");
        //把tensor放到向量中做运算
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor_img);
        //模型推理
        auto out=model.forward(inputs).toTensor();
        std::cout<<torch::argmax(out,1)<<std::endl;

    } catch (const c10::Error &e) {
        std::cerr<<e.what();
        return -1;
    }
}
