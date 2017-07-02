// testVector.cpp : �������̨Ӧ�ó������ڵ㡣
//


#include "stdafx.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
	//cv::namedWindow("hello");
	int size = 12;
	std::vector<cv::Point> x; 
	cv::Mat m(1, size, CV_8U, cv::Scalar(128));
	cv::findNonZero(m, x); // �ھ������ҵ������Ԫ��  ���������(Point)����(Vector)
	printf("findNonZero x %zd " , x.size() );
	//cv::waitKey(); // ���û��namedWindow��������  ��仰����ȴ�
	system("pause");
	return 0;
}
 /*
	 Mat rawImg(600, 500, CV_8UC3, Scalar(255, 0, 0));   Scalar����
	 // ����һ��600x500���ص�ͼ�� ÿ��Ԫ����8UC3(3��channelÿ��channel��8bit��ʾ)  ��ʼ��Ϊ(255,0,0) ������ɫͼ��  
	
	Scalar����ɴ��1��4����ֵ����ֵ
	typedef struct Scalar
	{
		double val[4];
	}Scalar;

	���ʹ�õ�ͼ����1ͨ���ģ���s.val[0]�д洢����
	���ʹ�õ�ͼ����3ͨ���ģ���s.val[0]��s.val[1]��s.val[2]�д洢����

*/