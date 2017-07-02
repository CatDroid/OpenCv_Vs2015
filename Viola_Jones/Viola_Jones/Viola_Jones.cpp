// Viola_Jones.cpp : �������̨Ӧ�ó������ڵ㡣
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

//���ļ�������OpenCV��װĿ¼�µ�\sources\data\haarcascades�ڣ���Ҫ����xml�ļ����Ƶ���ǰ����Ŀ¼��  
CascadeClassifier face_cascade;
void detectAndDisplay(Mat frame);

/*
OpenCV�����������

ʹ��OpenCV����������⣨Viola-Jones������ⷽ����

Haar������Ϊ���ࣺ��Ե�������������������������ͶԽ�����������ϳ�����ģ�塣

�ŵ㣺
1.����ͼ��integral image�����ټ���Haar-like������
2.����Adaboost�㷨��������ѡ��ͷ�����ѵ����������������ϳ�ǿ��������
3.���÷������������Ч�ʡ�



���ط�����
����xml��ʽ��ģ���ļ�������haarcascade_frontalface_atl.xml��haarcascade_frontalface_atl2.xmlЧ���Ϻ�
�ļ���OpenCV��װĿ¼�µġ�data/haarcascades/��·���¡�

*/
int main(int argc, char** argv) {
	cv::Mat image;
	image = imread("lena.jpg", 1);  //��ǰ���̵�imageĿ¼�µ�mm.jpg�ļ���ע��Ŀ¼����  
	printf("dims %d \n " , image.dims );

	assert(!image.empty());
	if (!face_cascade.load(face_cascade_name)) {
		printf("�������������󣬿���δ�ҵ��ļ����������ļ�������Ŀ¼�£�\n");
		return -1;
	}
	detectAndDisplay(image); //����������⺯��  
	waitKey(0);
	//��ͣ��ʾһ�¡�  
}

void detectAndDisplay(cv::Mat face) {
	std::vector<cv::Rect> faces;
	cv::Mat face_gray;
	cv::Mat face_gray_temp;

	cv::cvtColor(face, face_gray_temp, CV_BGR2GRAY);		//rgb����ת��Ϊ�Ҷ�����  
	equalizeHist(face_gray_temp , face_gray);			//ֱ��ͼ���⻯ 



	cv::Mat face_binary ;
	cv::threshold(face_gray, face_binary , 128 /*��ֵ*/, 255 /*���ֵ*/, cv::ThresholdTypes::THRESH_BINARY );

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
	face_cascade.detectMultiScale(face_gray, faces, // ��߶ȼ�⺯��   OpenCV�ɴ��� cvHaarDetectObjects���� 
		1.1,  // ���ű��� �������1 
		2,    // �ϲ�����ʱ��С��neighbor ÿ����ѡ�������ٰ����ĸ���Ԫ�ظ��� ???
		CV_HAAR_SCALE_IMAGE, // ����� ֻ�Ծɸ�ʽ�ķ�������Ч
							 // CANNY��Ե��� CV_HAAR_FIND_BIGGEST_OBJECTѰ����Ŀ�� CV_HAAR_DO_ROUGH_SEACH��������
		Size(1, 1));
	//face_cascade.detectMultiScale(face_binary, faces, 1.1, 2, cv::CASCADE_SCALE_IMAGE, Size(1, 1)); 

	printf("faces num %zd \n" , faces.size() );


	for (int i = 0; i < faces.size(); i++) {
		Point center(  faces[i].x + faces[i].width*0.5,   faces[i].y + faces[i].height*0.5   ); // Բ����
		ellipse(face, center, 
					Size(faces[i].width*0.5, faces[i].height*0.5), /* ��Բ ��������뾶 */
					0, 
					0, 360, /*��ʼ����ֹ�Ƕ�*/
					Scalar(255, 0, 0)/*��ɫ*/, 
					2/*��ϸ*/, 
					7/*������*/, 0);
	}

	imshow("����ʶ��", face);
	
}

#else 


#include "opencv/cv.h"			// C�ӿ� 			
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
	IplImage* image = cvLoadImage(filename, 1); // ������ imread ���� Mat image ���Ƿ��� IplImage 

	if (image)
	{
		detect_and_draw(image);
		cvWaitKey(0);
		cvReleaseImage(&image); // �ͷ�IplImage
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
	IplImage* small_img = cvCreateImage(cvSize(cvRound(img->width / scale), cvRound(img->height / scale)), 8, 1);// ԭ����/1.2
	cvCvtColor(img, gray, CV_BGR2GRAY);
	cvResize(gray, small_img, CV_INTER_LINEAR); // ����  ��СͼƬ
	cvEqualizeHist(small_img, small_img);		//ֱ��ͼ����  

	CvMemStorage* storage = 0;
	storage = cvCreateMemStorage(0); // 0 default 64k
	cvClearMemStorage(storage);
	double t = (double)cvGetTickCount();

	//const char* cascade_name = "haarcascade_frontalface_alt.xml"; /*    "haarcascade_profileface.xml";*/
	const char* cascade_name = "haarcascade_frontalface_alt2.xml";

	// ȷ���ļ�����
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
		0/*CV_HAAR_DO_CANNY_PRUNING*/,// CANNY��Ե��� CV_HAAR_FIND_BIGGEST_OBJECTѰ����Ŀ�� CV_HAAR_DO_ROUGH_SEACH��������
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