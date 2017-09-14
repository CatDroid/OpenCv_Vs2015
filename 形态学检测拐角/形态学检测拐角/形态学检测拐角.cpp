// 形态学检测拐角.cpp : 定义控制台应用程序的入口点。
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
	Mat img, gray;
	img = imread("border_test.jpg", 1);
	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(5, 5), 0);
	printf("gray : row %d col %d ch %d \n", gray.rows, gray.cols, gray.channels());
	Mat canny_gray;
	Canny(gray, canny_gray, 0, 50);			// Canny边缘检测
 
	// 构造5×5的结构元素，分别为十字形、菱形、方形和X型
	Mat cross = getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
	Mat diamond = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));// 菱形结构(实心)元素的定义稍麻烦一些
	diamond.at<uint8_t>(0, 0) = 0;
	diamond.at<uint8_t>(0, 1) = 0;
	diamond.at<uint8_t>(1, 0) = 0;
	diamond.at<uint8_t>(4, 4) = 0;
	diamond.at<uint8_t>(4, 3) = 0;
	diamond.at<uint8_t>(3, 4) = 0;
	diamond.at<uint8_t>(4, 0) = 0;
	diamond.at<uint8_t>(4, 1) = 0;
	diamond.at<uint8_t>(3, 0) = 0;
	diamond.at<uint8_t>(0, 3) = 0;
	diamond.at<uint8_t>(0, 4) = 0;
	diamond.at<uint8_t>(1, 4) = 0;
	Mat	square = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	Mat x =  getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
	

	Mat result1;												// dilate 之前是三通道 之后也是三通道 建议用灰度图单通道
	dilate(gray, result1, cross);								// 使用cross膨胀 原图像
	Mat resultDiamond;
	erode(result1, resultDiamond, diamond); result1.release();	// 使用菱形腐蚀图像

	
	dilate(gray, result1, x);									// 使用X膨胀 原图像
	Mat resultSquare;
	erode(result1, resultSquare, square); result1.release();	// 使用方形腐蚀图像

	// 将两幅闭运算的图像相减获得角
	absdiff(resultSquare, resultDiamond, result1); resultSquare.release(); resultDiamond.release();
	// 使用阈值获得二值图
	Mat result ;
	threshold(result1, result, 40, 255, THRESH_BINARY); result1.release();

	imshow("角点位置", result);
#if 0
	//circle(img, Point(5, 200), 5, Scalar(255, 0, 0));
	Scalar color(255, 0, 0);
	printf("row %d col %d ch %d \n", result.rows , result.cols, result.channels() );
	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.cols; j++) {
			if (result.at<uint8_t>(i, j) == 255) {
				circle(img, Point(j , i ) , 10 , color );
			}
		}
	}

	imshow("原图角点", img );
#endif 

	waitKey(0);
	return 0;
}

