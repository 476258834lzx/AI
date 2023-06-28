cmake_minimum_required(VERSION 3.15)
project(Opencv)
set(CMAKE_CXX_STANDARD 14)
add_executable(Opencv hist.cpp)

set(OpenCV_DIR "D:/lib/opencv/build/x64/vc15/lib")#设置opencv编译的路径
find_package(OpenCV REQUIRED)#查找opencv需要的其他库
target_link_libraries(Opencv ${OpenCV_LIBS})#链接编译时需要的opencv的动态库到当前的项目

#将动态链接库拷贝到当前程序所在文件夹
if (MSVC)
    file(GLOB OPENCV_DLLS "D:/lib/opencv/build/x64/vc15/bin/*.dll")
    add_custom_command(TARGET Opencv#目标路径
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS}
            $<TARGET_FILE_DIR:Opencv>#项目路径
            )
endif ()
