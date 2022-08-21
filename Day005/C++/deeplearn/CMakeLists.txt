cmake_minimum_required(VERSION 3.17)
project(deeplearn)

set(CMAKE_PREFIX_PATH  "D:/lib/libtorch")#设置libtorch编译的路径
set(OpenCV_DIR "D:/lib/opencv/build/x64/vc15/lib")#设置opencv编译的路径

find_package(Torch REQUIRED)#查找libtorch需要的其他库
find_package(OpenCV REQUIRED)#查找opencv需要的其他库
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(deeplearn main.cpp)
target_link_libraries(deeplearn ${TORCH_LIBRARIES} ${OpenCV_LIBS})#链接编译时需要的opencv的动态库到当前的项目
set_property(TARGET deeplearn PROPERTY CXX_STANDARD 14)

#将动态链接库拷贝到当前程序所在文件夹
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET deeplearn#目标路径
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:deeplearn>#项目路径
            )
    file(GLOB OPENCV_DLLS "D:/lib/opencv/build/x64/vc15/bin/*.dll")
    add_custom_command(TARGET deeplearn#目标路径
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS}
            $<TARGET_FILE_DIR:deeplearn>#项目路径
            )
endif ()
