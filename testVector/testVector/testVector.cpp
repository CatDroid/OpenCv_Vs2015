// testVector.cpp : 定义控制台应用程序的入口点。
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
	cv::findNonZero(m, x); // 在矩阵中找到非零的元素  返回坐标点(Point)数组(Vector)
	printf("findNonZero x %zd " , x.size() );
	//cv::waitKey(); // 如果没有namedWindow创建窗口  这句话不会等待
	system("pause");
	return 0;
}
 /*
	 Mat rawImg(600, 500, CV_8UC3, Scalar(255, 0, 0));   Scalar标量
	 // 生成一副600x500像素的图像 每个元素是8UC3(3个channel每个channel用8bit表示)  初始化为(255,0,0) 整个红色图像  
	
	Scalar定义可存放1—4个数值的数值
	typedef struct Scalar
	{
		double val[4];
	}Scalar;

	如果使用的图像是1通道的，则s.val[0]中存储数据
	如果使用的图像是3通道的，则s.val[0]，s.val[1]，s.val[2]中存储数据

*/