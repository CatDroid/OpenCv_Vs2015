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

class OpencvHOGDetctor {
public:
	OpencvHOGDetctor() {}

	inline int det(const cv::Mat& src_img) {
		if (src_img.empty())
			return 0;

		cv::HOGDescriptor hog;
		hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
		std::vector<cv::Rect> found, found_filtered;
		hog.detectMultiScale(src_img, found, 0, cv::Size(8, 8), cv::Size(32, 32),1.05, 2);
		size_t i, j;
		for (i = 0; i < found.size(); i++) {
			cv::Rect r = found[i];
			for (j = 0; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}

		for (i = 0; i < found_filtered.size(); i++) {
			cv::Rect r = found_filtered[i];
			r.x += cvRound(r.width * 0.1);
			r.width = cvRound(r.width * 0.8);
			r.y += cvRound(r.height * 0.06);
			r.height = cvRound(r.height * 0.9);
			cv::rectangle(src_img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
		}
		mResultMat = src_img;
		// cv::imwrite(path, mResultMat);
		std::cout << "det ends" << std::endl; 
		mRets = found_filtered;
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
	if (src_img.empty()) {
		std::cout << "find not exists " << std::endl;
		system("pause");
		return -1;
	}
	int size = detor.det(src_img);
	std::cout << "size = " << size << std::endl;

	std::vector<cv::Rect> result = detor.getResult();
	
	std::vector<cv::Rect>::iterator rect = result.begin();
	while (rect != result.end() ) { // 红色  本来 cv::HOGDescriptor 就已经画了
		cv::rectangle(src_img, *(rect++), cv::Scalar(0, 0, 255), 1 /*thickness*/, 8 /*lineType*/);
	}

	char buf[256]; memset(buf, 0, 256 );
	sprintf(buf,"人脸个数 %d" , size );
	cv::imshow(buf , src_img);

	cv::waitKey();

    return 0;
}

