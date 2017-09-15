// ������ȡ.cpp : �������̨Ӧ�ó������ڵ㡣
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
		vector<vector<Point>> contours;	// ÿ�������������
		vector<Vec4i> hierarchy;		// �����Ĳ㼶��ϵ
		// CV_RETR_EXTERNAL		ֻ�������
		// CHAIN_APPROX_SIMPLE	ֻ�����յ������
		// ��ȡĿ�������ĺ���
		
		double t = (double)getTickCount();
		findContours(image, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
		t = ((double)getTickCount() - t) / getTickFrequency(); //���ʱ�䣬��λ����
		printf("findContours cost %f s \n" , t );
		Scalar color(255, 0, 0);
		for (int i = 0; i < contours.size(); i++)
		{
			//contours[i]������ǵ�i��������contours[i].size()������ǵ�i�����������е����ص���  
			for (int j = 0; j < contours[i].size(); j++)
			{
				//���Ƴ�contours���������е����ص�  
				circle(imageSource, contours[i][j], 5, color);
			}

			//���hierarchy��������  
			//char ch[256];
			//sprintf(ch, "%d", i);
			//string str = ch;
			//cout << "����hierarchy�ĵ�" << str << " ��Ԫ������Ϊ��" << endl << hierarchy[i] << endl << endl;

			//��������  
			//drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
		}
		 
	}
	imshow("������ȡ", imageSource);

	waitKey(0);
    return 0;
}

