// OpenCV_OCL.cpp : 定义控制台应用程序的入口点。
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

#define USING_OPENCL 1
int main()
{
	{
		double matrix[3][2] = { { 3.0f , 4.0f },{ 2.0f , -2.0f },{ -5.0f, -4.0f } };
		cv::Mat test1(cv::Size(2, 3), CV_64FC1, matrix); // 注意 行列   cv::Size(宽/列数，高/行数)	
		std::cout << test1 << std::endl;
	}
	{
		cv::Mat a = (cv::Mat_<float>(3, 2) << 1, -1, 1, -1, 2, 0);
		cv::Mat b = (cv::Mat_<float>(2, 2) << 2, 4, 1, 3);
		cv::Mat c = a*b;
		std::cout << c << std::endl;
	}
	if(false)
	{
		cv::Mat image;
		image = imread("lena.jpg");
		printf("Image        %d \n", image.channels());
		printf("Image dims   %d\n ", image.dims);
		printf("Image cols   %d\n ", image.cols);
		printf("Image rows   %d\n ", image.rows);
		printf("Image data   %p\n ", image.data);
		printf("Image channel%d\n ", image.channels());
		printf("Image type   %d\n", image.type());
		assert(image.data != NULL);
	}

#if USING_OPENCL == 1 
 
	ocl::setUseOpenCL(true);
	int64 start = 0, end = 0;
	start = getTickCount();
	UMat img, gray ,gaussian,candy ; 
	img = imread("lena.jpg", CV_LOAD_IMAGE_COLOR ).getUMat(ACCESS_READ);
	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gaussian, Size(7, 7), 1.5);
	Canny(gaussian, candy, 0, 50);



	imshow(ocl::useOpenCL() ? "edges with OCL" : "edges not OCL"  , candy);

	end = getTickCount();
	printf("time: %f ms\n", 1000.0*(end - start) / getTickFrequency());
	waitKey(0);
	destroyAllWindows();
#else
	int64 start = 0, end = 0;
	start = getTickCount();
	Mat img, gray;
	img = imread("lena.jpg", 1);
	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(7, 7), 1.5);
	Canny(gray, gray, 0, 50);


	imshow("canny", gray);
	end = getTickCount();
	printf("time: %f ms\n", 1000.0*(end - start) / getTickFrequency());

	

#endif 
    return 0;
}

