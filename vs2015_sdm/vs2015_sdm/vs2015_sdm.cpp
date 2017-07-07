// vs2015_sdm.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <vector>
#include <iostream>


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "modelcfg.h"
#include "ldmarkmodel.h"

using namespace std;
using namespace cv;

extern int train_model();
extern int test_model_with_camera();
extern int test_model_with_jpeg();

int main()
{
	{
		std::vector<float> test = { 1.0, 2.0, 3.0 ,-4.0 };
		double norm1 = cv::norm(test, CV_L1); //  街区距离L1  L1范数是指向量中各个元素绝对值之和  稀疏规则算子  Lasso regularization
		double norm2 = cv::norm(test, CV_L2); //  欧氏距离L2  L2范数是指向量各元素的平方和然后求平方根  岭回归 Ridge Regression   权值衰减weight decay
	}
	{
		double matrix[3][2] = { { 3.0f , 4.0f },{ 2.0f , -2.0f },{ -5.0f, -4.0f } };
		cv::Mat test1(cv::Size(2, 3), CV_64FC1, matrix); // 注意 行列   cv::Size(宽/列数，高/行数)	
		cout << "test  = " << endl << test1 << endl << endl;
		//cv::Mat d = (cv::Mat_<double>(5, 6) << 1, 0, 1, 1, 2, 0, 0, 1, 0, 0, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 0, 0, 0, 1, 1, 0, 2);
		double norm1 = cv::norm(test1, CV_L1);// 20	  矩阵时候 不是 矩阵的一二阶范数  还是上面的定义  所有元素绝对值之和
		double norm2 = cv::norm(test1, CV_L2);// 8.6	跟matlab不一样											  所有元素平方和再开二次方	
											  //	列和范数，A每一列元素绝对值之和的最大值 matlab调用函数norm(A, 1) 
											  //	A'A矩阵的最大特征值的开平方  A'是转置矩阵  A的最大奇异值 欧几里德范数 谱范数  matlab调用函数norm(x, 2) 
	}

	test_model_with_jpeg();
    return 0;
}

int train_model()
{
	std::vector<ImageLabel> mImageLabels;
	if (!load_ImageLabels("mImageLabels-train.bin", mImageLabels)) {
		mImageLabels.clear();
		ReadLabelsFromFile(mImageLabels);
		save_ImageLabels(mImageLabels, "mImageLabels-train.bin");
	}
	std::cout << "训练数据一共有: " << mImageLabels.size() << std::endl;


	vector<vector<int>> LandmarkIndexs;
	vector<int> LandmarkIndex1(IteraLandmarkIndex1, IteraLandmarkIndex1 + LandmarkLength1);
	LandmarkIndexs.push_back(LandmarkIndex1);
	vector<int> LandmarkIndex2(IteraLandmarkIndex2, IteraLandmarkIndex2 + LandmarkLength2);
	LandmarkIndexs.push_back(LandmarkIndex2);
	vector<int> LandmarkIndex3(IteraLandmarkIndex3, IteraLandmarkIndex3 + LandmarkLength3);
	LandmarkIndexs.push_back(LandmarkIndex3);
	vector<int> LandmarkIndex4(IteraLandmarkIndex4, IteraLandmarkIndex4 + LandmarkLength4);
	LandmarkIndexs.push_back(LandmarkIndex4);
	vector<int> LandmarkIndex5(IteraLandmarkIndex5, IteraLandmarkIndex5 + LandmarkLength5);
	LandmarkIndexs.push_back(LandmarkIndex5);

	vector<int> eyes_index(eyes_indexs, eyes_indexs + 4);
	Mat mean_shape(1, 2 * LandmarkPointsNum, CV_32FC1, mean_norm_shape);
	//vector<HoGParam> HoGParams{{ VlHogVariant::VlHogVariantUoctti, 5, 11, 4, 1.0f },{ VlHogVariant::VlHogVariantUoctti, 5, 10, 4, 0.7f },{ VlHogVariant::VlHogVariantUoctti, 5, 8, 4, 0.4f },{ VlHogVariant::VlHogVariantUoctti, 5, 6, 4, 0.25f } };
	vector<HoGParam> HoGParams{ { VlHogVariant::VlHogVariantUoctti, 4, 11, 4, 0.9f },{ VlHogVariant::VlHogVariantUoctti, 4, 10, 4, 0.7f },{ VlHogVariant::VlHogVariantUoctti, 4, 9, 4, 0.5f },{ VlHogVariant::VlHogVariantUoctti, 4, 8, 4, 0.3f },{ VlHogVariant::VlHogVariantUoctti, 4, 6, 4, 0.2f } };
	vector<LinearRegressor> LinearRegressors(5);

	ldmarkmodel model(LandmarkIndexs, eyes_index, mean_shape, HoGParams, LinearRegressors);
	model.train(mImageLabels);
	save_ldmarkmodel(model, "PCA-SDM-model.bin");


	system("pause");
	return 0;
}



int test_model_with_camera()
{
	/*********************
	std::vector<ImageLabel> mImageLabels;
	if(!load_ImageLabels("mImageLabels-test.bin", mImageLabels)){
	mImageLabels.clear();
	ReadLabelsFromFile(mImageLabels, "labels_ibug_300W_test.xml");
	save_ImageLabels(mImageLabels, "mImageLabels-test.bin");
	}
	std::cout << "测试数据一共有: " <<  mImageLabels.size() << std::endl;
	*******************/

	ldmarkmodel modelt;
	std::string modelFilePath = "roboman-landmark-model.bin";
	while (!load_ldmarkmodel(modelFilePath, modelt)) {
		std::cout << "文件打开错误，请重新输入文件路径." << std::endl;
		std::cin >> modelFilePath;
	}

	cv::VideoCapture mCamera(0);
	if (!mCamera.isOpened()) {
		std::cout << "摄像头打开失败..." << std::endl;
		system("pause");
		return 0;
	}
	cv::Mat Image;
	cv::Mat current_shape;
	for (;;) {
		mCamera >> Image;
		modelt.track(Image, current_shape);
		cv::Vec3d eav;
		modelt.EstimateHeadPose(current_shape, eav);
		modelt.drawPose(Image, current_shape, 50);

		int numLandmarks = current_shape.cols / 2;
		for (int j = 0; j<numLandmarks; j++) {
			int x = current_shape.at<float>(j);
			int y = current_shape.at<float>(j + numLandmarks);
			std::stringstream ss;
			ss << j;
			//            cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
			cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
		}
		cv::imshow("Camera", Image);
		if (27 == cv::waitKey(5)) {
			mCamera.release();
			cv::destroyAllWindows();
			break;
		}
	}

	system("pause");
	return 0;
}

#define FACE_TRACE(...)  { printf(__VA_ARGS__); printf("\n");}
int test_model_with_jpeg()
{
	ldmarkmodel modelt;
	modelt.loadFaceDetModelFile("haar_roboman_ff_alt2.xml");
	//  lbpcascade_frontalface.xml  haarcascade_frontalface_alt2.xml

	std::string modelFilePath = "roboman-landmark-model.bin";
	if (!load_ldmarkmodel(modelFilePath, modelt)) { // 根据bin文件初始化 ldmarkmodel实例的参数
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
	[1]	同上
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
	int ret = modelt.track(image, current_shape); // 主要实现
	FACE_TRACE("track ret:%d", ret);

	cv::Vec3d eav;
	modelt.EstimateHeadPose(current_shape, eav);
	FACE_TRACE("pitch,yaw,roll:(%f,%f,%f)", eav[0], eav[1], eav[2]);
	//modelt.drawPose(Image, current_shape, 50);
	int numLandmarks = current_shape.cols / 2;

	cv::Scalar color(0, 255, 0); // B G R 
	for (int j = 0; j<numLandmarks; j++) {
		//faceInfo[2 * j] = ;
		//faceInfo[2 * j + 1] = ;
		//FACE_TRACE("faceInfo %f %f.",  current_shape.at<float>(j) ,  current_shape.at<float>(j + numLandmarks) );
		cv::circle(image,
			cv::Point(current_shape.at<float>(j), current_shape.at<float>(j + numLandmarks)),
			2,
			color);
	}

	cv::imshow("feature", image);
	cv::waitKey();

	//system("pause");
	return 0;
}
