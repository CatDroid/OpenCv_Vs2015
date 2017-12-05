// 颜色空间_色系转换.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/ocl.hpp"
using namespace cv;

#include <vector>
#include <iostream>
using namespace std;


int main()
{

	Mat imageRGB = imread("2.jpg", IMREAD_COLOR);

	Mat imageHSV;
	cvtColor(imageRGB, imageHSV, CV_BGR2HSV);//将RGB色系转为HSV色系

	printf("channel %d cols %d rows %d depth 0x%x \n" , 
		imageHSV.channels() , imageHSV.cols, imageHSV.rows, imageHSV.depth() );

	imshow("rgb", imageRGB);
	imshow("hsv", imageHSV);


	std::vector<cv::Mat> MatArrays;
	cv::split(imageHSV, MatArrays );//分离三个通道  
 
	imshow("hue", MatArrays[0]);
	imshow("saturation", MatArrays[1]);
	imshow("value", MatArrays[2]);

	printf("MatArrays[0] channel %d cols %d rows %d depth 0x%x \n",
		MatArrays[0].channels(), MatArrays[0].cols, MatArrays[0].rows, MatArrays[0].depth());
	printf("MatArrays[1] channel %d cols %d rows %d depth 0x%x \n",
		MatArrays[1].channels(), MatArrays[1].cols, MatArrays[1].rows, MatArrays[1].depth());
	printf("MatArrays[2] channel %d cols %d rows %d depth 0x%x \n",
		MatArrays[2].channels(), MatArrays[2].cols, MatArrays[2].rows, MatArrays[2].depth());

	Mat afterCanny;
	cv::Canny(MatArrays[1], afterCanny,  10, 200 );
	imshow("afterCanny", afterCanny );

	waitKey(0);

    return 0;
}

