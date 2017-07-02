// face_detect.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "ldmarkmodel.h"

#define FACE_TRACE(...)  { printf(__VA_ARGS__); printf("\n");}

using namespace std;
using namespace cv;

int main()
{
	ldmarkmodel modelt;
	modelt.loadFaceDetModelFile("haar_roboman_ff_alt2.xml");
	//  lbpcascade_frontalface.xml  haarcascade_frontalface_alt2.xml

	std::string modelFilePath = "roboman-landmark-model.bin";
	if (!load_ldmarkmodel(modelFilePath, modelt)) { // ����bin�ļ���ʼ�� ldmarkmodelʵ���Ĳ���
		FACE_TRACE("load_ldmarkmodel failed...");
		return -1;
	}
	/*
		meanShape  dims 2  rows 1 cols 136 
		LinearRegressors capacity 5
			  LinearRegresso[0] weights			Mat	dims 2  rows  3073  cols 136	data notNULL
								eigenvectors	Mat dims 0  rows  0			0			NULL
								meanvalue		Mat		 2		  0			0			NULL
								x				Mat		 2		  0			0			NULL
								isPCA			false 
							[1]							 
							[2]
							[3]
							[4] 
		 
	*/

	cv::Mat image;
	image = imread("lena.jpg");
	FACE_TRACE("Image %d ", image.channels());
	FACE_TRACE("Image dims %d ", image.dims);
	FACE_TRACE("Image cols %d ", image.cols);
	FACE_TRACE("Image rows %d ", image.rows);
	FACE_TRACE("Image data %p ", image.data);
	FACE_TRACE("Image channel %d ", image.channels());
	assert(image.data != NULL);

	cv::Mat current_shape;
	//for(int i=0; i<10; i++){
	FACE_TRACE("track start...");
	int ret = modelt.track(image, current_shape); // ��Ҫʵ��
	FACE_TRACE("track ret:%d", ret);

	cv::Vec3d eav;
	modelt.EstimateHeadPose(current_shape, eav);
	FACE_TRACE("pitch,yaw,roll:(%f,%f,%f)", eav[0], eav[1], eav[2]);
	//modelt.drawPose(Image, current_shape, 50);
	int numLandmarks = current_shape.cols / 2;

	cv::Scalar color(0, 255, 0 ); // B G R 
	for (int j = 0; j<numLandmarks; j++) {
		//faceInfo[2 * j] = ;
		//faceInfo[2 * j + 1] = ;
		//FACE_TRACE("faceInfo %f %f.", 
		//				current_shape.at<float>(j) , 
		//				current_shape.at<float>(j + numLandmarks) );
		cv::circle(image,
					cv::Point(current_shape.at<float>(j) , current_shape.at<float>(j + numLandmarks)),
					2,
					color );
	}

	cv::imshow("������",image);
	cv::waitKey();

	//system("pause");

    return 0;
}

