// svm_test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"


#include "opencv2/core/core.hpp"			 // Basic OpenCV structures (cv::Mat, Scalar)
#include "opencv2/objdetect/objdetect.hpp"  // CascadeClassifier
#include "opencv2/highgui/highgui.hpp"		// OpenCV window I/O
#include "opencv2/imgproc/imgproc.hpp"		
#include "opencv2/ml/ml.hpp"				// SVM

using namespace cv;
using namespace cv::ml;

int main()
{
	int width = 512, height = 512;                      //512*512 的正方形区域  
	Mat image = Mat::zeros(height, width, CV_8UC3);

	int labels[8] = { 1, 1, 0, 0, 1, 1, 0, 0 };          //8 个结果 (标签) 
	Mat labelsMat(8, 1, CV_32SC1, labels);

	float trainingData[8][2] = { { 10, 10 },             //8 样本点（和标签对应）  
	{ 10, 50 },
	{ 501, 255 },
	{ 500, 501 },
	{ 40,30 },
	{ 70, 60 },
	{ 300,300 },
	{ 60, 500 } };
	Mat trainingDataMat(8, 2, CV_32FC1, trainingData);

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);            // 类型  
	svm->setKernel(SVM::LINEAR);     	//  核函数  
	Ptr<TrainData> td = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat); //样本是按行排列的  

	svm->train(td);      //训练  

	Vec3b green(0, 255, 0), blue(255, 0, 0);
	Mat sampleMat(1, 2, CV_32F);
	float response;
	// 预测512*512正方形区域内的每个点的归类  
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			sampleMat.at<float>(0, 0) = i;
			sampleMat.at<float>(0, 1) = j;
			response = svm->predict(sampleMat);

			if (response == 1)      //1画绿色  
				image.at<Vec3b>(i, j) = green;
			else if (response == 0) //0画蓝色  
				image.at<Vec3b>(i, j) = blue;
		}

	// 标出样本点的位置  
	int thickness = -1;
	int lineType = 8;
	int x, y;
	Scalar s;
	for (int i = 0; i < 8; i++) {
		if (labels[i]) {
			s = Scalar(255, 0, 255);
		}
		else {
			s = Scalar(255, 255, 0);
		}
		x = trainingData[i][0];
		y = trainingData[i][1];
		circle(image, Point(x, y), 5, s, thickness, lineType);
	}


	imshow("SVM Simple Example", image);
	waitKey(0);
}

