// pyramid.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "stdio.h"

using namespace std;
using namespace cv;


enum pyrType { 
	PYR_GUASS, 
	PYR_LAPLACE
};

void genPyr(const Mat& imgSrc, vector<Mat>& outPutArray, int TYPE, int level)
{
	outPutArray.assign(level + 1, Mat());
	outPutArray[0] = imgSrc.clone(); // the 0 level is the image. 
	for (int i = 0; i != level; i++)
	{
		pyrDown(outPutArray[i], outPutArray[i + 1]);			 // 图像下采样 
	}// 滤波器类型 选择的是 高斯核 CV_GAUSSIAN_5x5 
	if (PYR_GUASS == TYPE)
	{
		return;
	}
	for (int i = 0; i != level; i++)
	{
		Mat UpSampleImg;
		pyrUp(						// 图像上采样 
			outPutArray[i + 1],		// 金字塔上一层 
			UpSampleImg, 
			outPutArray[i].size()	// 扩大成 金字塔本层大小 
					); 
		outPutArray[i]/*金字塔本层*/ -= UpSampleImg; /*金字塔上一层的上采样*/
	}
}

/*
	(高斯/拉普拉斯)图像金字塔 高斯核 
	http://www.cnblogs.com/ronny/p/3886013.html
*/
int main()
{
	Mat src = imread("lena.jpg");
	 
		vector<Mat> output1; 
		genPyr(src , output1, PYR_GUASS , 3 );

		for (int i = 0; i < output1.size() ; i++) { // 高斯金字塔
			char buf[256];
			sprintf(buf, "Ga Level %d ", i );
			imshow(buf , output1[i] );
		}

		waitKey();
		cv::destroyAllWindows();
	 

	 
		vector<Mat> output;
		genPyr(src, output, PYR_LAPLACE, 3);

		for (int i = 0; i < output.size(); i++) {
			char buf[256];
			sprintf(buf, "LP Level %d ", i);
			imshow(buf, output[i]); // 拉普拉斯金字塔 N=0底层  顶层是原层高斯图
		}
		// 它的每一层图像 是 高斯金字塔本层图像 与 其高一级的图像经内插放大后图像  的差 
		// 此过程相当于带通滤波 因此拉普拉斯金字塔又称为 带通金字塔分解

		waitKey();
		cv::destroyAllWindows();
	 

		for (int i = 0; i < output.size(); i++) {
			char buf[256];
			sprintf(buf, "Level %d ", i);
			imshow(buf, output[i] + output1[i]); // 
		}
		waitKey();
		cv::destroyAllWindows();

    return 0;
}

