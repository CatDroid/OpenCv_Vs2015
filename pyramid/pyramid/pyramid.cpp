// pyramid.cpp : �������̨Ӧ�ó������ڵ㡣
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
		pyrDown(outPutArray[i], outPutArray[i + 1]);			 // ͼ���²��� 
	}// �˲������� ѡ����� ��˹�� CV_GAUSSIAN_5x5 
	if (PYR_GUASS == TYPE)
	{
		return;
	}
	for (int i = 0; i != level; i++)
	{
		Mat UpSampleImg;
		pyrUp(						// ͼ���ϲ��� 
			outPutArray[i + 1],		// ��������һ�� 
			UpSampleImg, 
			outPutArray[i].size()	// ����� �����������С 
					); 
		outPutArray[i]/*����������*/ -= UpSampleImg; /*��������һ����ϲ���*/
	}
}

/*
	(��˹/������˹)ͼ������� ��˹�� 
	http://www.cnblogs.com/ronny/p/3886013.html
*/
int main()
{
	Mat src = imread("lena.jpg");
	 
		vector<Mat> output1; 
		genPyr(src , output1, PYR_GUASS , 3 );

		for (int i = 0; i < output1.size() ; i++) { // ��˹������
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
			imshow(buf, output[i]); // ������˹������ N=0�ײ�  ������ԭ���˹ͼ
		}
		// ����ÿһ��ͼ�� �� ��˹����������ͼ�� �� ���һ����ͼ���ڲ�Ŵ��ͼ��  �Ĳ� 
		// �˹����൱�ڴ�ͨ�˲� ���������˹�������ֳ�Ϊ ��ͨ�������ֽ�

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

