#include "opencvtest.h"

using namespace std;
using namespace cv;

/*
OpenCV3.2/sources/samples/cpp/train_HOG.cpp
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size )
也是实现可视化HOG
From http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
*/
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu


/*
	HOGDescriptor visual_imagealizer
	adapted for arbitrary size of feature sets and training images
	适应任何大小的特征集和训练图片
	http://download.csdn.net/detail/u011285477/9472067
*/
Mat get_hogdescriptor_visual_image(Mat& origImg,
	vector<float>& descriptorValues,//hog特征向量
	Size winSize,//图片窗口大小
	Size cellSize,             
	int scaleFactor,//缩放背景图像的比例
	double viz_factor)//缩放hog特征的线长比例
{   
	Mat visual_image;//最后可视化的图像大小
	resize(origImg, visual_image, Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));

	int gradientBinSize = 9;
	// dividing 180° into 9 bins, how large (in rad) is one bin?
	float radRangeForOneBin = 3.14/(float)gradientBinSize; //pi=3.14对应180°

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;//x方向上的cell个数
	int cells_in_y_dir = winSize.height / cellSize.height;//y方向上的cell个数
	int totalnrofcells = cells_in_x_dir * cells_in_y_dir;//cell的总个数
	//注意此处三维数组的定义格式
	//int ***b;
	//int a[2][3][4];
	//int (*b)[3][4] = a;
	//gradientStrengths[cells_in_y_dir][cells_in_x_dir][9]
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter   = new int*[cells_in_y_dir];
	for (int y=0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x=0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin=0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;//把每个cell的9个bin对应的梯度强度都初始化为0
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	//相当于blockstride = (8,8)
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky=0; blocky<blocks_in_y_dir; blocky++)            
		{
			// 4 cells per block ...
			for (int cellNr=0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				int cellx = blockx;
				int celly = blocky;
				if (cellNr==1) celly++;
				if (cellNr==2) cellx++;
				if (cellNr==3)
				{
					cellx++;
					celly++;
				}

				for (int bin=0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[ descriptorDataIdx ];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;//因为C是按行存储

				} // for (all bins)


				// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;//由于block之间有重叠，所以要记录哪些cell被多次计算了

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	// compute average gradient strengths
	for (int celly=0; celly<cells_in_y_dir; celly++)
	{
		for (int cellx=0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin=0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	
	cout << "winSize = " << winSize << endl;
	cout << "cellSize = " << cellSize << endl;
	cout << "blockSize = " << cellSize*2<< endl;
	cout << "blockNum = " << blocks_in_x_dir<<"×"<<blocks_in_y_dir << endl;
	cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

	// draw cells
	for (int celly=0; celly<cells_in_y_dir; celly++)
	{
		for (int cellx=0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize.width;
			int drawY = celly * cellSize.height;

			int mx = drawX + cellSize.width/2;
			int my = drawY + cellSize.height/2;

			rectangle(visual_image,
				Point(drawX*scaleFactor,drawY*scaleFactor),
				Point((drawX+cellSize.width)*scaleFactor,
				(drawY+cellSize.height)*scaleFactor),
				CV_RGB(0,0,0),//cell框线的颜色
				1);

			// draw in each cell all 9 gradient strengths
			for (int bin=0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength==0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;//取每个bin里的中间值，如10°,30°,...,170°.

				float dirVecX = cos( currRad );
				float dirVecY = sin( currRad );
				float maxVecLen = cellSize.width/2;
				float scale = viz_factor; // just a visual_imagealization scale,
				// to see the lines better

				// compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visual_imagealization
				line(visual_image,
					Point(x1*scaleFactor,y1*scaleFactor),
					Point(x2*scaleFactor,y2*scaleFactor),
					CV_RGB(255,255,255),//HOG可视化的cell的颜色
					1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	// don't forget to free memory allocated by helper data structures!
	for (int y=0; y<cells_in_y_dir; y++)
	{
		for (int x=0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];            
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visual_image;//返回最终的HOG可视化图像

}


int main()
{
	 
	HOGDescriptor hog;//使用的是默认的hog参数  也可以通过HOGDescriptor::load(const String& filename, const String& objname)加载XML文件初始化 下面参数 还有 vector<float>SVMDetector 参数
	/*
	HOGDescriptor(
		Size win_size=Size(64, 128),	//	检测窗口大小
		Size block_size=Size(16, 16),	//	Block大小
		Size block_stride=Size(8, 8),	//	Block每次移动宽度包括水平和垂直两个方向  
		Size cell_size=Size(8, 8),		//	Cell单元大小  block_size./cell_size = 一个block中有多少cell
		int nbins=9,					//	直方图bin数目 
		int _derivAperture=1			//	?? 
		double win_sigma=DEFAULT_WIN_SIGMA(DEFAULT_WIN_SIGMA=-1),
										//	高斯函数的方差  默认-1 
										//  默认情况实际是(blockSize.width + blockSize.height)/8  确保block取值在2*sigma之间 
		 int _histogramNormType=HOGDescriptor::L2Hys,
										//	直方图归一化类型(最后这个检测窗口的一维特征向量 归一化)
		double threshold_L2hys=0.2,		//	L2Hys化中限制最大值为0.2  归一化后不能超过0.2 

		bool gamma_correction=true,		//  是否Gamma校正 
		int nlevels=DEFAULT_NLEVELS,
		bool _signedGradient= false		//  bin角度在0~360(true) 还是0~180(false)
		)

	Parameters:	
	win_size – Detection window size. Align to block size and block stride.
	block_size – Block size in pixels. Align to cell size. Only (16,16) is supported for now.
	block_stride – Block stride. It must be a multiple of cell size.
	cell_size – Cell size. Only (8, 8) is supported for now.
	nbins – Number of bins. Only 9 bins per cell are supported for now.
	win_sigma – Gaussian smoothing window parameter.
	threshold_L2hys – L2-Hys normalization method shrinkage.
	gamma_correction – Flag to specify whether the gamma correction preprocessing is required or not.
	nlevels – Maximum number of detection window increases.
	*/
	// 对于 128*80 的图片，blockstride = 8, 15*9 的 block，  9*15 * 2*2 *9 = 4860
	//			block_x_num = ( 128 - 16 ) / 8 + 1 =  15 
	//			block_y_num = ( 80  - 16 ) / 8 + 1 =  9 
	//			

	//int width = 80;
	//int height = 128;
	//hog.winSize=Size(width,height); // 直接使用图片的大小
	vector<float> des;//HOG特征向量
	Mat src = imread("objimg.jpg");
	int width =  src.cols;
	int height = src.rows;
	std::cout << "channels : " << src.channels() << std::endl;
	std::cout << "cols : " << src.cols << std::endl;
	std::cout << "elemSize1 : " << src.elemSize1() << std::endl;// 一个元素的一个通道占的字节数(数据类型!)
	std::cout << "elemSize : " << src.elemSize() << std::endl;// 3 每个元素占字节数  = channel * elemSize1 (float double uchar )
	std::cout << "step : " << src.step  << std::endl;		// 237  step[0]  图片是cols=79 rows=128 elemSize=3 79*3 = 237
	std::cout << "step[0] : " << src.step[0] << std::endl; // 237		step[0]		一行占多少字节
	std::cout << "step[1] : " << src.step[1] << std::endl; // 3			step[1]		一个元素占多少字节 == Mat::elemSize()
	std::cout << "step[2] : " << src.step[2] << std::endl; // N/A(dim=2) 二维无效
	std::cout << "Mat.step.size_t() " << size_t(src.step) << std::endl; // 237 step[0]

	/*
		如果dim=3 三维 
		step[0] = 面的大小(字节数)
		step[1] = 行的大小(字节数)
		step[2] = 元素的大小(字节数)
	
	*/

	Mat& dst = src  ;
	//resize(src,dst,Size(width,height));//规范图像尺寸
	imshow("src",src);
	hog.compute(
		dst,
		des ,
		cv::Size(8,8)/*检测窗口移动的步进 默认=cellSize */, 
		cv::Size(8,8)/*在原图像加上多少个像素 padding*2 */
	);//计算hog特征
	Mat background = Mat::zeros(Size(width,height),CV_8UC1);//设置黑色背景图，因为要用白色绘制hog特征

	std::cout << "hot feature diamens = " << des.size() << std::endl;
	/*
	virtual void compute(
			InputArray img,							// in 待检测或计算的图像
			CV_OUT std::vector<float>& descriptors, // out Hog描述结构
			Size winStride = Size(),				// in 窗口移动步伐 如果不指定的话 就是 winStride = cellSize;
			Size padding = Size(),					// in 扩充图像相关尺寸
			const std::vector<Point>& locations = std::vector<Point>()
													// in 对于正样本可以直接取(0,0),负样本为随机产生合理坐标范围内的点坐标
			) const;


	CV_Assert(blockSize.width % cellSize.width == 0 &&
			blockSize.height % cellSize.height == 0);
	CV_Assert((winSize.width - blockSize.width) % blockStride.width == 0 &&
			(winSize.height - blockSize.height) % blockStride.height == 0 );

	src:
		rows	128	int
		cols	79	int
	HOGDescriptor:
	+		winSize		{width=64 height=128 }	cv::Size_<int>
	+		blockSize	{width=16 height=16 }	cv::Size_<int>
	+		blockStride	{width=8 height=8 }		cv::Size_<int>
	+		cellSize	{width=8 height=8 }		cv::Size_<int>
			nbins	9	int
			winStride = cellSize

	HOG特征向量 7560  = (2) * ( 7*15 * (2*2) * 9  ) 
						|		|		|     |_一个cell的特征向量维数=nbins
						|		|		|_一个block的cell数目
						|		|_ 一个window的block数目 (winSize.w-blockSize.w)/blockStride.w + 1  
						|_ 一整图中 window的数目 需要根据参数 winStride和padding 还有原图的宽高 
								( 79 - 64 ) / 8  + 1 =  1(1.875) + 1 = 2  宽有 0.875*8=7个像素被拆去
								( from HOGCache::windowsInImage(const Size& imageSize, const Size& winStride)  )
								 imageSize 是被padded过的 默认是0
								 如果HOGDescriptor.compute(.. Size padding = Size(8,8) ..)
								 那么图片大小 79,128 --> 79+8*2,128+8*2  ?? 为什么padding要乘2  
								窗口数目 = [(79+8*2-64)/8+1]  * [(128+8*2-128)/8+1] = 4 * 3 = 12  
								整个图像特征点 12 * ( 7*15 * (2*2) * 9  ) = 45360 



	HOGDescriptor::detectMultiScale  使用HOG特征进行多尺度检测
		HOGInvoker(...)
			hog->detect  --> void HOGDescriptor::detect(const Mat& img, ....

	HOGCache
	HOGInvoker
	HOGDescriptor

	int gcd(int a,int b)//最大公约数Greatest Common Divisor
							辗转相除法求最大公约数的简写算法，也称欧几里德算法
	*/

	Mat d = get_hogdescriptor_visual_image(background,des,hog.winSize,hog.cellSize,3,2.5);
	imshow("dst",d);
	imwrite("hogvisualize.jpg",d);
	waitKey();

	 return 0;
}
