// Viola_Jones.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#if 1
#include "opencv2/core/core.hpp"			 // Basic OpenCV structures (cv::Mat, Scalar)
#include "opencv2/objdetect/objdetect.hpp"  // CascadeClassifier
#include "opencv2/highgui/highgui.hpp"		// OpenCV window I/O
#include "opencv2/imgproc/imgproc.hpp"		

#include <iostream>   
#include <stdio.h>   

using namespace std;
using namespace cv;
// string face_cascade_name = "haar_roboman_ff_alt2.xml";'
//string face_cascade_name = "lbpcascade_frontalface.xml";
 string face_cascade_name = "haarcascade_frontalface_alt.xml";

//该文件存在于OpenCV安装目录下的\sources\data\haarcascades内，需要将该xml文件复制到当前工程目录下  
CascadeClassifier face_cascade;
void detectAndDisplay(Mat frame);

/*
OpenCV用于人脸检测

使用OpenCV进行人脸检测（Viola-Jones人脸检测方法）

Haar特征分为三类：边缘特征，线性特征，中心特征和对角线特征，组合成特征模板。

优点：
1.积分图像（integral image）快速计算Haar-like特征。
2.利用Adaboost算法进行特征选择和分类器训练，把弱分类器组合成强分类器。
3.采用分类器级联提高效率。



加载分类器
读入xml格式的模型文件，其中haarcascade_frontalface_atl.xml和haarcascade_frontalface_atl2.xml效果较好
文件在OpenCV安装目录下的“data/haarcascades/”路径下。

*/
int main(int argc, char** argv) {
	cv::Mat image;
	image = imread("lena.jpg", 1);  //当前工程的image目录下的mm.jpg文件，注意目录符号  
	printf("dims %d \n " , image.dims );

	assert(!image.empty());
	//  CascadeClassifier为级联分类器检测类
	// 使用Adaboost的方法
	// 提取LBP\HOG\HAAR特征进行目标检测
	// 用例: 加载traincascade.cpp进行训练的分类器

	if (!face_cascade.load(face_cascade_name)) {
		printf("级联分类器错误，可能未找到文件，拷贝该文件到工程目录下！\n");
		return -1;
	}
	detectAndDisplay(image); //调用人脸检测函数  
	waitKey(0);
	//暂停显示一下。  
}

void detectAndDisplay(cv::Mat face) {
	std::vector<cv::Rect> faces;
	cv::Mat face_gray;
	cv::Mat face_gray_temp;

	cv::cvtColor(face, face_gray_temp, CV_BGR2GRAY);		//rgb类型转换为灰度类型  
	equalizeHist(face_gray_temp , face_gray);			//直方图均衡化 



	cv::Mat face_binary ;
	cv::threshold(face_gray, face_binary , 128 /*阀值*/, 255 /*最大值*/, cv::ThresholdTypes::THRESH_BINARY );

	cv::namedWindow("before_hist");
	cv::imshow("before_hist", face_gray_temp);
	cv::namedWindow("after_hist");
	cv::imshow("after_hist", face_gray);
	cv::namedWindow("face_binary");
	cv::imshow("face_binary", face_binary);

	cv::waitKey();
	cv::destroyAllWindows();

	printf("chl %d row %d col %d\n " , face_gray.channels() , face_gray.rows,face_gray.cols );
	//face_cascade.detectMultiScale(face_gray, faces, 1.1, 2, CASCADE_SCALE_IMAGE, cv::Size(40, 40) );// robot
	face_cascade.detectMultiScale(face_gray, faces, // 多尺度检测函数   OpenCV旧代码 cvHaarDetectObjects函数 
		1.1,  // 缩放比例 必须大于1   每次图像尺寸减小的比例为1.1   原图宽/1.1/1.1/1.1 ... 
		2,    // 表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸)
		CV_HAAR_SCALE_IMAGE,	// 检测标记 只对旧格式的分类器有效
								// CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像
								// CANNY边缘检测 CV_HAAR_FIND_BIGGEST_OBJECT寻找最目标 CV_HAAR_DO_ROUGH_SEACH初略搜索
		Size(1, 1)				// 最小检测窗口大小 目标的 最小最大尺寸
								// 最大检测窗口大小（默认是图像大小）
	);
	/*
		http://blog.csdn.net/u011447369/article/details/52451144
		xml文件里面width和height是分类器样本尺寸
		ex4：
		分类器：18*18；输入图像：26*26；detectMultiScale(frame_gray,vct_rc, 1.05, 1, 0, Size(22,22))；
		多尺度检测尺寸：21*21、20*20、19*19

	*/ 

	//face_cascade.detectMultiScale(face_binary, faces, 1.1, 2, cv::CASCADE_SCALE_IMAGE, Size(1, 1)); 

	printf("faces num %zd \n" , faces.size() );


	for (int i = 0; i < faces.size(); i++) {
		Point center(  faces[i].x + faces[i].width*0.5,   faces[i].y + faces[i].height*0.5   ); // 圆中心
		ellipse(face, center, 
					Size(faces[i].width*0.5, faces[i].height*0.5), /* 椭圆 两个主轴半径 */
					0, 
					0, 360, /*起始和终止角度*/
					Scalar(255, 0, 0)/*颜色*/, 
					2/*粗细*/, 
					7/*线类型*/, 0);
	}

	imshow("人脸识别", face);
	
}

#else 


#include "opencv/cv.h"			// C接口 			
#include "opencv/highgui.h"				 

#include "opencv2/objdetect/objdetect.hpp"  // CascadeClassifier

#include <stdio.h>   
#include <stdlib.h>   
#include <string.h>   
#include <assert.h>   
#include <math.h>   
#include <float.h>   
#include <limits.h>   
#include <time.h>   
#include <ctype.h>  



void detect_and_draw(IplImage* image);

int main(int argc, char** argv)
{

	
	cvNamedWindow("result", 1);

	const char* filename = "lena.jpg";
	IplImage* image = cvLoadImage(filename, 1); // 不是用 imread 返回 Mat image 而是返回 IplImage 

	if (image)
	{
		detect_and_draw(image);
		cvWaitKey(0);
		cvReleaseImage(&image); // 释放IplImage
	}

	cvDestroyWindow("result");

	return 0;
}


void detect_and_draw(IplImage* img)
{
	double scale = 1.2;
	static CvScalar colors[8] = {
		{ 0,0,255 } ,	{ 0,128,255 } ,		{ 0,255,255 } ,		{ 0,255,0 } ,
		{ 255,128,0 }, { 255,255,0 }  ,		{ 255,0,0 }  ,	 { 255,0,255 }  
	};//Just some pretty colors to draw with  

	// CvSize CvScalar

	//Image Preparation    
	IplImage* gray = cvCreateImage(cvSize(img->width, img->height), 8, 1); // 8bit 1channnel
	IplImage* small_img = cvCreateImage(cvSize(cvRound(img->width / scale), cvRound(img->height / scale)), 8, 1);// 原来的/1.2
	cvCvtColor(img, gray, CV_BGR2GRAY);
	cvResize(gray, small_img, CV_INTER_LINEAR); // 线性  缩小图片
	cvEqualizeHist(small_img, small_img);		//直方图均衡  

	CvMemStorage* storage = 0;
	storage = cvCreateMemStorage(0); // 0 default 64k
	cvClearMemStorage(storage);
	double t = (double)cvGetTickCount();

	//const char* cascade_name = "haarcascade_frontalface_alt.xml"; /*    "haarcascade_profileface.xml";*/
	const char* cascade_name = "haarcascade_frontalface_alt2.xml";

	// 确保文件存在
	FILE* f = fopen(cascade_name, "rb");
	if (!f) {
		printf("ERROR fopen !\n");
		return  ;
	}
	fseek(f, 0, SEEK_END);
	int size = ftell(f) + 1;
	fclose(f);
	printf("file ok %d\n", size);

	//CvHaarClassifierCascade* cascade = 0;
	cv::CascadeClassifier cascade;

	try {
		// cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);
		if (  !  cascade.load(cascade_name)  ) {
			fprintf(stderr, "ERROR: 1 load classifier cascade Exception \n" );
			return  ;
		}
	}
	catch (cv::Exception ex) {
		fprintf(stderr, "ERROR: 2 load classifier cascade Exception %s \n", ex.msg.c_str());
		system("pause");
		return ;
	}

 

	CvSeq* objects = cvHaarDetectObjects(small_img,
		(CvHaarClassifierCascade*)cascade.getOldCascade(),
		storage,
		1.1,
		2,
		0/*CV_HAAR_DO_CANNY_PRUNING*/,// CANNY边缘检测 CV_HAAR_FIND_BIGGEST_OBJECT寻找最目标 CV_HAAR_DO_ROUGH_SEACH初略搜索
		cvSize(30, 30));

	t = (double)cvGetTickCount() - t;
	printf("detection time = %gms\n", t / ((double)cvGetTickFrequency()*1000.));

	//Loop through found objects and draw boxes around them   
	for (int i = 0; i<(objects ? objects->total : 0); ++i)
	{
		CvRect* r = (CvRect*)cvGetSeqElem(objects, i);
		cvRectangle(img, cvPoint(r->x*scale, r->y*scale), cvPoint((r->x + r->width)*scale, (r->y + r->height)*scale), colors[i % 8]);
	}
	for (int i = 0; i < (objects ? objects->total : 0); i++)
	{
		CvRect* r = (CvRect*)cvGetSeqElem(objects, i);
		CvPoint center;
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);
		cvCircle(img, center, radius, colors[i % 8], 3, 8, 0);
	}

	cvShowImage("result", img);
	cvReleaseImage(&gray);
	cvReleaseImage(&small_img);
	cvReleaseMemStorage(&storage);
}

#endif 