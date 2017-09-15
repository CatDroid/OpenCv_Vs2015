// 轮廓提取.cpp : 定义控制台应用程序的入口点。
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
	// 0 = IMREAD_GRAYSCALE
	// 1 = IMREAD_COLOR
	Mat imageSource = imread("2.jpg", IMREAD_COLOR);
	Mat image;
	imshow("Source Image", imageSource);
	cvtColor(imageSource, image, COLOR_BGR2GRAY);

	GaussianBlur(image, image, Size(3, 3), 0);
	Canny(image, image, 10, 250);

	printf("channel %d \n", image.channels() );

	imshow("canny",image);

	//if (false) 
	{
		vector<vector<Point>> contours;	// 每个轮廓的坐标点
		vector<Vec4i> hierarchy;		// 轮廓的层级关系
		// CV_RETR_EXTERNAL		只保留外层
		// CHAIN_APPROX_SIMPLE	只保留拐点的坐标
		// 提取目标轮廓的函数
		
		double t = (double)getTickCount();
		findContours(image, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
		t = ((double)getTickCount() - t) / getTickFrequency(); //获得时间，单位是秒
		printf("findContours cost %f s \n" , t );
		Scalar color(255, 0, 0);
		for (int i = 0; i < contours.size(); i++)
		{
			//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数  
			for (int j = 0; j < contours[i].size(); j++)
			{
				//绘制出contours向量内所有的像素点  
				circle(imageSource, contours[i][j], 5, color);
			}

			//输出hierarchy向量内容  
			//char ch[256];
			//sprintf(ch, "%d", i);
			//string str = ch;
			//cout << "向量hierarchy的第" << str << " 个元素内容为：" << endl << hierarchy[i] << endl << endl;

			//绘制轮廓  
			//drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
		}
		 
	}
	imshow("轮廓提取", imageSource);

	waitKey(0);
    return 0;
}

