// pedestrian_det_withOpenCV.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <vector>

// copy from dlib-android	jni_pedestrian_det.cpp 
class OpencvHOGDetctor {
public:
	OpencvHOGDetctor() {}

	inline int det(const cv::Mat& src_img) {
		if (src_img.empty())
			return 0;

		cv::HOGDescriptor hog;
		hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
		std::vector<cv::Rect> found, found_filtered;
		std::cout << "Default Hog Descriptor Size  " << hog.getDescriptorSize() << std::endl;
		cv::Mat feature;
		hog.compute(src_img , &feature);
		hog.detectMultiScale(src_img, found, 0, cv::Size(8, 8), cv::Size(32, 32)  ,1.05, 2);
		std::cout << "After Hog Descriptor Size  " << hog.getDescriptorSize() << std::endl; // 3780
		// double hit_threshold		程序内部计算为行人目标的阈值，也就是检测到的特征到SVM分类超平面的距离 
		// Size win_stride=Size()	滑动窗口每次移动的距离。它必须是块移动的整数倍
		// Size padding=Size()		图像扩充的大小
		// double scale0=1.05		为比例系数，即被检测图像每一次被压缩的比例

		/*
			HOG，是目前计算机视觉、模式识别领域很常用的一种描述图像'局部纹理的特征'

			winSize(64,128),blockSize(16,16),blockStride(8,8)步进,cellSize(8,8)
			窗口大小64x128,块大小16x16，块步长8x8，那么窗口中块的数目是(（64-16）/8+1)*((128-16)/8+1) = 7*15 =105个块，
			块大小为16x16,胞元大小为8x8，那么一个块中的胞元cell数目是 (16/8)*(16/8) = 4个胞元

			n= 105 x 4 x 9 = 3780

			OpenCV计算HOG描述子的维数
			size_t HOGDescriptor::getDescriptorSize()const{
				return (size_t)nbins*
						(blockSize.width/cellSize.width)*(blockSize.height/cellSize.height)* 
						((winSize.width - blockSize.width)/blockStride.width + 1)* ((winSize.height - blockSize.height)/blockStride.height + 1);
			}

		*/
		size_t i, j;

		// 找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中 
		for (i = 0; i < found.size(); i++) {
			cv::Rect r = found[i];
			for (j = 0; j < found.size(); j++)
				if (j != i && (r & found[j]) == r) // cv::Rect 两个矩形的交集和并集  rect1 & rect2  rect1 | rect2
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}

		// 画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整  
		for (i = 0; i < found_filtered.size(); i++) {
			cv::Rect r = found_filtered[i];
			r.x +=		cvRound(r.width  * 0.1); // 这里缩小了框 
			r.width =	cvRound(r.width  * 0.8);
			r.y +=		cvRound(r.height * 0.06);
			r.height =	cvRound(r.height * 0.9);
			cv::rectangle(src_img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2); // 画图
		}
		mResultMat = src_img;
		// cv::imwrite(path, mResultMat);
		std::cout << "det ends" << std::endl; 
		mRets = found_filtered; // 返回的是没有嵌套的框
		return found_filtered.size();
	}

	inline cv::Mat& getResultMat() { return mResultMat; }

	inline std::vector<cv::Rect>& getResult() { return mRets; }

private:
	cv::Mat mResultMat;
	std::vector<cv::Rect> mRets;
};

int main()
{
	OpencvHOGDetctor detor;
	cv::Mat src_img = cv::imread("pets.jpg" , CV_LOAD_IMAGE_COLOR /*==1 */);
	// cv::cvtColor(rgbaMat, bgrMat, cv::COLOR_RGBA2BGR); dlib-android中图片解码后是RGBA 所以还需要转成BGR
	if (src_img.empty()) {
		std::cout << "find not exists " << std::endl;
		system("pause");
		return -1;
	}
	int size = detor.det(src_img);
	std::cout << "rect size = " << size << std::endl;
	

	std::vector<cv::Rect> result = detor.getResult();
	
	std::vector<cv::Rect>::iterator rect = result.begin(); 
	while (rect != result.end() ) { 
		// 我们画的是红色    OpencvHOGDetctor内部就已经画了(绿色的)
		// 返回的就是 hog检测出的矩形 ，在 OpencvHOGDetctor内部绘制的时候 (绿色的), 会缩小矩形框!
		cv::rectangle(src_img, *(rect++), cv::Scalar(0, 0, 255), 1 /*thickness*/, 8 /*lineType*/);
	}

	char buf[256]; memset(buf, 0, 256 );
	sprintf(buf,"人脸个数 %d" , size );
	cv::imshow(buf , src_img);

	cv::waitKey();

	/*
	OpenCV中包含了2种HOG的实现途径，一种是HOG+SVM的实现方法，另一种是HOG+Cascade的实现方法

	vector<Rect> found1, found_filtered1,found2, found_filtered2;//矩形框数组
	//方法1，HOG+SVM
	
	HOGDescriptor hog;//HOG特征检测器  
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//设置SVM分类器为默认参数     
	hog.detectMultiScale(src, found1, 0, Size(2, 2), Size(0, 0), 1.05, 2);//对图像进行多尺度检测，检测窗口移动步长为(8,8)  
 
	//方法2.HOG+Cascade 
 
	CascadeClassifier *cascade = new CascadeClassifier;
	cascade->load("hogcascade_pedestrians.xml");
	cascade->detectMultiScale(src, found2);

	http://blog.csdn.net/qq_14845119/article/details/52187774
	文章最后实验:
	Hog+Svm		(红色): 检测率高 检测时间长
	Hog+Cascade (绿色): 检测率低 检测时间短(实时)

	矩形区间（R-HOG）和环形区间（C-HOG）
	R-HOG区间 方形的格子 三个参数来表征：每个区间中细胞单元的数目、每个细胞单元中像素点的数目、每个细胞的直方图通道数目

	*/

    return 0;
}

