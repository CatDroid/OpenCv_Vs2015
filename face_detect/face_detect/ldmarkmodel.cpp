
#include "stdafx.h"

#include "ldmarkmodel.h"

#define FACE_TRACE(...)  { printf(__VA_ARGS__ ); printf("\n");}
#define   LOG_BUF_LEN (1024*1)




template<class T = int>
static cv::Rect_<T> get_enclosing_bbox(cv::Mat landmarks)
{
    auto num_landmarks = landmarks.cols / 2;
    double min_x_val, max_x_val, min_y_val, max_y_val;
    cv::minMaxLoc(landmarks.colRange(0, num_landmarks), &min_x_val, &max_x_val);
    cv::minMaxLoc(landmarks.colRange(num_landmarks, landmarks.cols), &min_y_val, &max_y_val);
    double width  = max_x_val - min_x_val;
    double height =  max_y_val - min_y_val;
    return cv::Rect_<T>(min_x_val, min_y_val, width, height);
//    return cv::Rect_<T>(min_x_val, min_y_val, width, height);
}


/**
 * Performs an initial alignment of the model, by putting the mean model into
 * the center of the face box.
 *
 * An optional scaling and translation parameters can be given to generate
 * perturbations of the initialisation.
 *
 * Note 02/04/15: I think with the new perturbation code, we can delete the optional
 * parameters here - make it as simple as possible, don't include what's not needed.
 * Align and perturb should really be separate - separate things.
 *
 * @param[in] mean Mean model points.
 * @param[in] facebox A facebox to align the model to.
 * @param[in] scaling_x Optional scaling in x of the model.
 * @param[in] scaling_y Optional scaling in y of the model.
 * @param[in] translation_x Optional translation in x of the model.
 * @param[in] translation_y Optional translation in y of the model.
 * @return A cv::Mat of the aligned points.
 */
cv::Mat align_mean(cv::Mat mean, cv::Rect facebox, float scaling_x=1.0f, float scaling_y=1.0f, float translation_x=0.0f, float translation_y=0.0f)
{
    using cv::Mat;
	// 假定了脸部是正方形的 x和y都在[-0.5 , 0.5] 范围    为什么x加0.5  y加0.3  

    // Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
    // More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
    // if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
    Mat aligned_mean = mean.clone();
	 
    Mat aligned_mean_x = aligned_mean.colRange(0, aligned_mean.cols / 2); // 平均脸部  x坐标矩阵 
    Mat aligned_mean_y = aligned_mean.colRange(aligned_mean.cols / 2, aligned_mean.cols);// 平均脸部  y坐标矩阵 
    aligned_mean_x = (aligned_mean_x*scaling_x + 0.5f + translation_x) * facebox.width + facebox.x; // 矩阵运算   
    aligned_mean_y = (aligned_mean_y*scaling_y + 0.3f + translation_y) * facebox.height + facebox.y;
	// aligned_mean  和 aligned_mean_x 和 aligned_mean_y 都是指向同一片内存数据 data地址一样 但是rows和cols不一样 cv::MatStep cv::MatSize 不一样
	// 也就是colRange可以引用原来矩阵的某一部分 对其操作就会影响原来的矩阵

	//cv::Mat m1();
    return aligned_mean;
}


cv::Mat align_mean(cv::Mat mean, cv::Mat landmarks)
{
    using cv::Mat;

    static float scaling_x = 1.0f;
    static float scaling_y = 1.0f;
    static float translation_x = 0.0f;
    static float translation_y = 0.0f;
    static bool isFirstCalled = true;
    if(isFirstCalled){
        isFirstCalled = false;
        auto   num_landmarks = mean.cols / 2; // 特征点数目  Mat只有1行136列 68个特征点  前面68列是x 后面68列是y
        double min_x_val, max_x_val, min_y_val, max_y_val;
        cv::minMaxLoc(	mean.colRange(0, num_landmarks), //矩阵[0,68)列组成矩阵 		
						&min_x_val, &max_x_val  // 找到[0,68)列组成矩阵中的最大值 因为前面68列是x坐标 所以返回是x坐标的最大和最小值 
					); 
        cv::minMaxLoc(	mean.colRange(num_landmarks, mean.cols),// 矩阵[68,136)列组成矩阵 
						&min_y_val, &max_y_val
					);
        scaling_x = 1.0f/(max_x_val - min_x_val);
        scaling_y = 1.0f/(max_y_val - min_y_val);
        translation_x=0.0f;
        translation_y=0.0f;
    }

    static auto num_landmarks = landmarks.cols / 2;
    double min_x_val, max_x_val, min_y_val, max_y_val;
    cv::minMaxLoc(landmarks.colRange(0, num_landmarks), &min_x_val, &max_x_val);
    cv::minMaxLoc(landmarks.colRange(num_landmarks, landmarks.cols), &min_y_val, &max_y_val);
    double width  = max_x_val - min_x_val; // 这里得到 头像框 的宽和高   
    double height = max_y_val - min_y_val;

    // Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
    // More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
    // if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
    Mat aligned_mean = mean.clone();
    Mat aligned_mean_x = aligned_mean.colRange(0, aligned_mean.cols / 2);
    Mat aligned_mean_y = aligned_mean.colRange(aligned_mean.cols / 2, aligned_mean.cols);
    aligned_mean_x = (aligned_mean_x*scaling_x + 0.5f + translation_x) * width + min_x_val;
    aligned_mean_y = (aligned_mean_y*scaling_y + 0.3f + translation_y) * height + min_y_val;
    return aligned_mean;
}


/**
 * Perturb by a certain x and y translation and an optional scaling.
 *
 * tx, ty are in percent of the total face box width/height.
 *
 * @param[in] facebox A facebox to align the model to.
 * @param[in] translation_x Translation in x of the box.
 * @param[in] translation_y Translation in y of the box.
 * @param[in] scaling Optional scale factor of the box.
 * @return A perturbed cv::Rect.
 */
cv::Rect perturb(cv::Rect facebox)
{
    float translation_x = (rand()%20-10)*0.01;
    float translation_y = (rand()%20-10)*0.01;
    float scaling = 1.0 + (rand()%20-10)*0.015;
//    cout << scaling << endl;
    auto tx_pixel = translation_x * facebox.width;
    auto ty_pixel = translation_y * facebox.height;
    // Because the reference point is on the top left and not in the center, we
    // need to temporarily store the new width and calculate half of the offset.
    // We need to move it further to compensate for the scaling, i.e. keep the center the center.
    auto perturbed_width = facebox.width * scaling;
    auto perturbed_height = facebox.height * scaling;
    //auto perturbed_width_diff = facebox.width - perturbed_width;
    //auto perturbed_height_diff = facebox.height - perturbed_height;
    // Note: Rounding?
    cv::Rect perturbed_box(facebox.x + (facebox.width - perturbed_width) / 2.0f + tx_pixel, facebox.y + (facebox.height - perturbed_height) / 2.0f + ty_pixel, perturbed_width, perturbed_height);

    return perturbed_box;
}




std::string trim(const std::string& str)
{
    std::string::size_type pos = str.find_first_not_of(' ');
    if (pos == std::string::npos)
    {
        return str;
    }
    std::string::size_type pos2 = str.find_last_not_of(' ');
    if (pos2 != std::string::npos)
    {
        return str.substr(pos, pos2 - pos + 1);
    }
    return str.substr(pos);
}

std::string replace(const std::string& str, const std::string& dest, const std::string& src)
{
    std::string ret = str;
    size_t pos = ret.find(dest);
    while(pos != std::string::npos){
        ret = ret.replace(pos, dest.length(), src);
        pos = ret.find(dest);
    }
    return ret;
}

std::vector<std::string> split(const  std::string& s, const std::string& delim)
{
    std::vector<std::string> elems;
    size_t pos = 0;
    size_t len = s.length();
    size_t delim_len = delim.length();
    if (delim_len == 0) return elems;
    while (pos < len)
    {
        int find_pos = s.find(delim, pos);
        if (find_pos < 0)
        {
            elems.push_back(s.substr(pos, len - pos));
            break;
        }
        elems.push_back(s.substr(pos, find_pos - pos));
        pos = find_pos + delim_len;
    }
    return elems;
}
/*
void ReadLabelsFromFile(std::vector<ImageLabel> &Imagelabels, std::string Path = "labels_ibug_300W_train.xml"){
    std::string ParentPath(trainFilePath);
    std::ifstream LabelsFile(ParentPath+Path, std::ios::in);
    if(!LabelsFile.is_open())
        return;
    std::string linestr;
    while(std::getline(LabelsFile, linestr)){
        linestr = trim(linestr);
        linestr = replace(linestr, "</", "");
        linestr = replace(linestr, "/>", "");
        linestr = replace(linestr, "<", "");
        linestr = replace(linestr, ">", "");
        linestr = replace(linestr, "'", "");

        std::vector<std::string> strNodes = split(linestr, " ");
        static ImageLabel* mImageLabel = NULL;
        switch (strNodes.size()) {
        case 1:
            if(strNodes[0] == "image"){
                Imagelabels.push_back(*mImageLabel);
                delete mImageLabel;
            }
            break;
        case 2:
            if(strNodes[0] == "image"){
                mImageLabel = new ImageLabel();
                mImageLabel->imagePath = ParentPath + split(strNodes[1], "=")[1];
//                std::cout << mImageLabel->imagePath << std::endl;
//                cv::Mat Image = cv::imread(mImageLabel->imagePath);
//                cv::imshow("Image", Image);
//                cv::waitKey(0);
            }
            break;
        case 5:
            if(strNodes[0] == "box"){
                mImageLabel->faceBox[0] = atoi(split(strNodes[1], "=")[1].data());
                mImageLabel->faceBox[1] = atoi(split(strNodes[2], "=")[1].data());
                mImageLabel->faceBox[2] = atoi(split(strNodes[3], "=")[1].data());
                mImageLabel->faceBox[3] = atoi(split(strNodes[4], "=")[1].data());
            }
            break;
        case 4:
            if(strNodes[0] == "part"){
                int index = atoi(split(strNodes[1], "=")[1].data());
                mImageLabel->landmarkPos[index] = atoi(split(strNodes[2], "=")[1].data());
                mImageLabel->landmarkPos[index+LandmarkPointsNum] = atoi(split(strNodes[3], "=")[1].data());
            }
            break;
        default:
            break;
        }
    }
    LabelsFile.close();
}
*/


bool load_ImageLabels(std::string filename, std::vector<ImageLabel> &mImageLabels)
{
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open())
        return false;
    cereal::BinaryInputArchive input_archive(file);
    input_archive(mImageLabels);
    return true;
}


void save_ImageLabels(std::vector<ImageLabel> mImageLabels, std::string filename)
{
    std::ofstream file(filename, std::ios::binary);
    cereal::BinaryOutputArchive output_archive(file);
    output_archive(mImageLabels);
}

cv::Mat CalculateHogDescriptor(cv::Mat &grayImage, cv::Mat shape, std::vector<int> LandmarkIndexs, std::vector<int> eyes_index, HoGParam mHoGParam){

    // This is in pixels in the original image:
    int numLandmarks = shape.cols/2;
    float lx = ( shape.at<float>(eyes_index.at(0))+shape.at<float>(eyes_index.at(1)) )*0.5;
    float ly = ( shape.at<float>(eyes_index.at(0)+numLandmarks)+shape.at<float>(eyes_index.at(1)+numLandmarks) )*0.5;
    float rx = ( shape.at<float>(eyes_index.at(2))+shape.at<float>(eyes_index.at(3)) )*0.5;
    float ry = ( shape.at<float>(eyes_index.at(2)+numLandmarks)+shape.at<float>(eyes_index.at(3)+numLandmarks) )*0.5;
    float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );
    int patch_width_half = round(mHoGParam.relative_patch_size * distance);

    cv::Mat hogDescriptors; // We'll get the dimensions later from vl_hog_get_*
    //prepare hog first
    int fixed_roi_size = mHoGParam.num_cells * mHoGParam.cell_size;
    VlHog* hog = vl_hog_new(VlHogVariant::VlHogVariantUoctti, mHoGParam.num_bins, false); // transposed (=col-major) = false
    vl_hog_prepare_buf(hog, fixed_roi_size, fixed_roi_size, mHoGParam.cell_size);
    int ww = static_cast<int>(vl_hog_get_width(hog)); // assert ww == hh == numCells
    int hh = static_cast<int>(vl_hog_get_height(hog));
    int dd = static_cast<int>(vl_hog_get_dimension(hog)); // assert ww=hogDim1, hh=hogDim2, dd=hogDim3
    cv::Mat hogArray(1, ww*hh*dd, CV_32FC1); // safer & same result. Don't use C-style memory management.

    for (int i = 0; i < LandmarkIndexs.size(); i++) {
        int x = cvRound(shape.at<float>(LandmarkIndexs.at(i)));
        int y = cvRound(shape.at<float>(LandmarkIndexs.at(i)+numLandmarks));

        cv::Mat roiImg;
        if (x - patch_width_half < 0 || y - patch_width_half < 0 || x + patch_width_half >= grayImage.cols || y + patch_width_half >= grayImage.rows) {
            // The feature extraction location is too far near a border. We extend the
            // image (add a black canvas) and then extract from this larger image.
            int borderLeft = (x - patch_width_half) < 0 ? std::abs(x - patch_width_half) : 0; // x and y are patch-centers
            int borderTop = (y - patch_width_half) < 0 ? std::abs(y - patch_width_half) : 0;
            int borderRight = (x + patch_width_half) >= grayImage.cols ? std::abs(grayImage.cols - (x + patch_width_half)) : 0;
            int borderBottom = (y + patch_width_half) >= grayImage.rows ? std::abs(grayImage.rows - (y + patch_width_half)) : 0;
            cv::Mat extendedImage = grayImage.clone();
            cv::copyMakeBorder(extendedImage, extendedImage, borderTop, borderBottom, borderLeft, borderRight, cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::Rect roi((x - patch_width_half) + borderLeft, (y - patch_width_half) + borderTop, patch_width_half * 2, patch_width_half * 2); // Rect: x y w h. x and y are top-left corner.
            roiImg = extendedImage(roi).clone(); // clone because we need a continuous memory block
        }
        else {
            cv::Rect roi(x - patch_width_half, y - patch_width_half, patch_width_half * 2, patch_width_half * 2); // x y w h. Rect: x and y are top-left corner. Our x and y are center. Convert.
            roiImg = grayImage(roi).clone(); // clone because we need a continuous memory block
        }
        // This has to be the same for each image, so each image's HOG descriptor will have the same dimensions, independent of the image's resolution
        int fixed_roi_size = mHoGParam.num_cells * mHoGParam.cell_size;
        cv::resize(roiImg, roiImg, cv::Size(fixed_roi_size, fixed_roi_size));

        roiImg.convertTo(roiImg, CV_32FC1); // vl_hog_put_image expects a float* (values 0.0f-255.0f)

        vl_hog_get_array(hog, (float*)roiImg.data, roiImg.cols, roiImg.rows,
                                    mHoGParam.cell_size, hogArray.ptr<float>(0));
        hogDescriptors.push_back(hogArray);
    }

    vl_hog_delete(hog);
    // concatenate all the descriptors for this sample vertically (into a row-vector):
    hogDescriptors = hogDescriptors.reshape(0, hogDescriptors.cols * hogDescriptors.rows).t();
    // add a bias row (affine part)
    cv::Mat bias = cv::Mat::ones(1, 1, CV_32FC1);
    cv::hconcat(hogDescriptors, bias, hogDescriptors);
    return hogDescriptors;
}

LinearRegressor::LinearRegressor() : weights(),meanvalue(),x(),isPCA(false)
{

}

bool LinearRegressor::learn(cv::Mat &data, cv::Mat &labels, bool isPCA)
{
    this->isPCA = isPCA;
    if(this->isPCA){
        cv::Mat mdata = data.colRange(0, data.cols-2).clone();
        cv::PCA FeaturePCA(mdata, cv::Mat(), CV_PCA_DATA_AS_ROW);
        std::cout << "特征向量尺度: " <<FeaturePCA.eigenvectors.size() << std::endl;
        std::cout << "特征值尺度:   " <<FeaturePCA.eigenvalues.size() << std::endl;
        double eigensum = cv::sum(FeaturePCA.eigenvalues)[0];
        double lamda = 0.0;
        int index = 0;
        for(int i=0; i<FeaturePCA.eigenvalues.rows; i++){
            lamda += FeaturePCA.eigenvalues.at<float>(i,0);
            if(lamda/eigensum > 0.97){
                index = i;
                std::cout << "特征个数可以压缩为:" << i << "个" << std::endl;
                break;
            }
        }
        this->meanvalue = FeaturePCA.mean;
        this->eigenvectors = FeaturePCA.eigenvectors.rowRange(0, index).t();
        for(int i=0; i<mdata.rows; i++){
            mdata.row(i) = mdata.row(i) - this->meanvalue;
        }
        mdata = mdata*this->eigenvectors;
        cv::Mat A = cv::Mat::zeros(mdata.rows, mdata.cols+1, mdata.type());
        for(int i=0; i<mdata.rows; i++){
            for(int j=0; j<mdata.cols; j++){
                A.at<float>(i,j) = mdata.at<float>(i,j);
            }
        }
        A.col(A.cols-1) = cv::Mat::ones(A.rows, 1, A.type());
        mdata.release();
        //自己的写的最小二乘
        cv::Mat AT = A.t();
        cv::Mat ATA = A.t()*A;
        float lambda = 1.50f * static_cast<float>(cv::norm(ATA)) / static_cast<float>(A.rows);
        cv::Mat regulariser = cv::Mat::eye(ATA.size(), ATA.type())*lambda;
        regulariser.at<float>(regulariser.rows-1, regulariser.cols-1) = 0.0f;
        this->x = (ATA + regulariser).inv(cv::DECOMP_LU)*AT*labels;
        //opencv提供的最小二乘
        //cv::solve(A, labels, this->x);

//            this->weights = this->eigenvectors*this->x;
//            this->eigenvectors.release();
    }else{
        cv::Mat A = data.clone();
        //自己的写的最小二乘
        cv::Mat AT = A.t();
        cv::Mat ATA = A.t()*A;
        float lambda = 1.50f * static_cast<float>(cv::norm(ATA)) / static_cast<float>(A.rows);
        cv::Mat regulariser = cv::Mat::eye(ATA.size(), ATA.type())*lambda;
        regulariser.at<float>(regulariser.rows-1, regulariser.cols-1) = 0.0f;
        this->weights = (ATA + regulariser).inv(cv::DECOMP_LU)*AT*labels;
        //opencv提供的最小二乘
        //cv::solve(A, labels, this->weights);
    }
    return true; // see todo above
}

double LinearRegressor::test(cv::Mat data, cv::Mat labels)
{
    cv::Mat predictions;
    for (int i = 0; i < data.rows; ++i) {
        cv::Mat prediction = this->predict(data.row(i));
        predictions.push_back(prediction);
    }
    return cv::norm(predictions, labels, cv::NORM_L2) / cv::norm(labels, cv::NORM_L2);
}


cv::Mat LinearRegressor::predict(cv::Mat values)
{
    if(this->isPCA){
        cv::Mat mdata = values.colRange(0, values.cols-2).clone();
//            assert(mdata.cols==this->weights.rows && mdata.cols==this->meanvalue.cols);
        if(mdata.rows == 1){
            mdata = (mdata - this->meanvalue)*this->eigenvectors;
            cv::Mat A = cv::Mat::zeros(mdata.rows, mdata.cols+1, mdata.type());
            for(int i=0; i<mdata.cols; i++){
                A.at<float>(i) = mdata.at<float>(i);
            }
            A.at<float>(A.cols-1) = 1.0f;
            return A*this->x;
        }
        else{
            for(int i=0; i<mdata.rows; i++){
                mdata.row(i) = mdata.row(i) - this->meanvalue;
            }
            mdata = mdata*this->eigenvectors;
            cv::Mat A = cv::Mat::zeros(mdata.rows, mdata.cols+1, mdata.type());
            for(int i=0; i<mdata.rows; i++){
                for(int j=0; j<mdata.cols; j++){
                    A.at<float>(i,j) = mdata.at<float>(i,j);
                }
            }
            A.col(A.cols-1) = cv::Mat::ones(A.rows, 1, A.type());
            return A*this->x;
        }
    }else{
        assert(values.cols==this->weights.rows);
        return  values*this->weights;
    }
}

void LinearRegressor::convert(std::vector<int> &tar_LandmarkIndex){
    if(isPCA){

    }else{
        assert(this->weights.cols/2 >= tar_LandmarkIndex.size());
        cv::Mat tmp = this->weights.clone();
        this->weights.release();
        this->weights.create(tmp.rows, tar_LandmarkIndex.size()*2, tmp.type());
        for(int i=0; i<this->weights.rows; i++){
            for(int j=0; j<tar_LandmarkIndex.size(); j++){
                this->weights.at<float>(i, j) = tmp.at<float>(i, tar_LandmarkIndex.at(j));
                this->weights.at<float>(i, j+tar_LandmarkIndex.size()) = tmp.at<float>(i, tar_LandmarkIndex.at(j)+tmp.cols/2);
            }
        }
        tmp.release();
    }
}

ldmarkmodel::ldmarkmodel(){
    //{36,39,42,45,30,48,54};   {7,16,17,8,9,10,11};
    static int HeadPosePointIndexs[] = {36,39,42,45,30,48,54};
    estimateHeadPosePointIndexs = HeadPosePointIndexs;
    static float estimateHeadPose2dArray[] = {
        -0.208764,-0.140359,0.458815,0.106082,0.00859783,-0.0866249,-0.443304,-0.00551231,-0.0697294,
        -0.157724,-0.173532,0.16253,0.0935172,-0.0280447,0.016427,-0.162489,-0.0468956,-0.102772,
        0.126487,-0.164141,0.184245,0.101047,0.0104349,-0.0243688,-0.183127,0.0267416,0.117526,
        0.201744,-0.051405,0.498323,0.0341851,-0.0126043,0.0578142,-0.490372,0.0244975,0.0670094,
        0.0244522,-0.211899,-1.73645,0.0873952,0.00189387,0.0850161,1.72599,0.00521321,0.0315345,
        -0.122839,0.405878,0.28964,-0.23045,0.0212364,-0.0533548,-0.290354,0.0718529,-0.176586,
        0.136662,0.335455,0.142905,-0.191773,-0.00149495,0.00509046,-0.156346,-0.0759126,0.133053,
        -0.0393198,0.307292,0.185202,-0.446933,-0.0789959,0.29604,-0.190589,-0.407886,0.0269739,
        -0.00319206,0.141906,0.143748,-0.194121,-0.0809829,0.0443648,-0.157001,-0.0928255,0.0334674,
        -0.0155408,-0.145267,-0.146458,0.205672,-0.111508,0.0481617,0.142516,-0.0820573,0.0329081,
        -0.0520549,-0.329935,-0.231104,0.451872,-0.140248,0.294419,0.223746,-0.381816,0.0223632,
        0.176198,-0.00558382,0.0509544,0.0258391,0.050704,-1.10825,-0.0198969,1.1124,0.189531,
        -0.0352285,0.163014,0.0842186,-0.24742,0.199899,0.228204,-0.0721214,-0.0561584,-0.157876,
        -0.0308544,-0.131422,-0.0865534,0.205083,0.161144,0.197055,0.0733392,-0.0916629,-0.147355,
        0.527424,-0.0592165,0.0150818,0.0603236,0.640014,-0.0714241,-0.0199933,-0.261328,0.891053};
    estimateHeadPoseMat = cv::Mat(15,9,CV_32FC1,estimateHeadPose2dArray);
    static float estimateHeadPose2dArray2[] = {
        0.139791,27.4028,7.02636,
        -2.48207,9.59384,6.03758,
        1.27402,10.4795,6.20801,
        1.17406,29.1886,1.67768,
        0.306761,-103.832,5.66238,
        4.78663,17.8726,-15.3623,
        -5.20016,9.29488,-11.2495,
        -25.1704,10.8649,-29.4877,
        -5.62572,9.0871,-12.0982,
        -5.19707,-8.25251,13.3965,
        -23.6643,-13.1348,29.4322,
        67.239,0.666896,1.84304,
        -2.83223,4.56333,-15.885,
        -4.74948,-3.79454,12.7986,
        -16.1,1.47175,4.03941 };
    estimateHeadPoseMat2 = cv::Mat(15,3,CV_32FC1,estimateHeadPose2dArray2);
    //loadFaceDetModelFile();
    FACE_TRACE("ldmarkmodel() No Paras Constructor  LinearRegressors size  %d " , LinearRegressors.size() );
}

ldmarkmodel::ldmarkmodel(std::vector<std::vector<int>> LandmarkIndexs, std::vector<int> eyes_index, cv::Mat meanShape, std::vector<HoGParam> HoGParams, std::vector<LinearRegressor> LinearRegressors) :
    LandmarkIndexs(LandmarkIndexs),eyes_index(eyes_index),meanShape(meanShape),HoGParams(HoGParams),isNormal(true),LinearRegressors(LinearRegressors)
{
    //loadFaceDetModelFile();
    FACE_TRACE("ldmarkmodel(...) Constructor  LinearRegressors size  %d " , LinearRegressors.size() );
}

void ldmarkmodel::loadFaceDetModelFile(std::string filePath){
    face_cascade.load(filePath);
    assert( !face_cascade.empty() )  ;
}

void ldmarkmodel::train(std::vector<ImageLabel> &mImageLabels){
    assert(HoGParams.size() >= LinearRegressors.size());
    int samplesNum = 800;    //mImageLabels.size()/10;
    std::cout << "请输入训练样本个数，最少20个." << std::endl;
    std::cin >> samplesNum;
    if(samplesNum < 20)
        samplesNum = 20;
    else if(samplesNum > mImageLabels.size())
        samplesNum = mImageLabels.size();
    std::cout << "一共" << samplesNum << "个训练样本.\n" << std::endl;

    std::cout << "是否以两眼距离标准归一化Landmarker坐标偏差?\n[Y/N?]" << std::endl;
    std::string t;
    std::cin >> t;
    if(t=="Y" || t=="y")
        isNormal = true;
    else if(t=="N" || t=="n")
        isNormal = false;
    if(isNormal)
        std::cout << "归一化坐标.\n" << std::endl;
    else
        std::cout << "不归一化坐标.\n" << std::endl;


    cv::Mat current_shape(samplesNum, meanShape.cols, CV_32FC1);
    cv::Mat target_shape(samplesNum, meanShape.cols, CV_32FC1);
//        cv::namedWindow("Image", cv::WINDOW_NORMAL);
    for(int i=0; i<samplesNum; i++){
        for(int j=0; j<meanShape.cols; j++){
            target_shape.at<float>(i, j)  = mImageLabels.at(i).landmarkPos[j];
        }
        cv::Mat Image = cv::imread(mImageLabels.at(i).imagePath, CV_LOAD_IMAGE_GRAYSCALE); //CV_LOAD_IMAGE_GRAYSCALE
        cv::Rect faceBox(mImageLabels.at(i).faceBox[0],mImageLabels.at(i).faceBox[1],mImageLabels.at(i).faceBox[2],mImageLabels.at(i).faceBox[3]);
        cv::Rect efaceBox = get_enclosing_bbox(target_shape.row(i));
        cv::Rect mfaceBox = perturb(faceBox) & cv::Rect(0,0,Image.cols, Image.rows);
        if((float)(efaceBox & faceBox).area()/faceBox.area()<0.4)
            mfaceBox = perturb(efaceBox) & cv::Rect(0,0,Image.cols, Image.rows);
        cv::Mat  align_shape = align_mean(meanShape, mfaceBox);
        assert(align_shape.rows == 1);
        for(int j=0; j<meanShape.cols; j++){
            current_shape.at<float>(i, j) = align_shape.at<float>(j);
        }

//            int numLandmarks = meanShape.cols/2;
//            for(int j=0; j<numLandmarks; j++){
//                int x = current_shape.at<float>(i, j);
//                int y = current_shape.at<float>(i, j + numLandmarks);
//                cv::circle(Image, cv::Point(x, y), 5, cv::Scalar(255, 0, 0), -1);
//                x = target_shape.at<float>(i, j);
//                y = target_shape.at<float>(i, j + numLandmarks);
//                cv::circle(Image, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
//            }
//            cv::rectangle(Image,  faceBox, cv::Scalar(0, 255, 0), 1, 4);
//            cv::rectangle(Image, mfaceBox, cv::Scalar(255, 0, 0), 3);
//            cv::rectangle(Image, efaceBox, cv::Scalar(0, 0, 255), 1, 4);
//            cv::imshow("Image", Image);
//            std::cout << (float)(efaceBox & faceBox).area()/faceBox.area() << std::endl;
//            if((float)(efaceBox & faceBox).area()/faceBox.area()<0.4)
//                cv::waitKey(0);
//            cv::waitKey(10);
    }
    float error0 = 0.0f;
    int numLandmarks = target_shape.cols/2;
    for(int i=0; i<samplesNum; i++){
        cv::Mat shape = current_shape.row(i);
        float lx = ( shape.at<float>(eyes_index.at(0))+shape.at<float>(eyes_index.at(1)) )*0.5;
        float ly = ( shape.at<float>(eyes_index.at(0)+numLandmarks)+shape.at<float>(eyes_index.at(1)+numLandmarks) )*0.5;
        float rx = ( shape.at<float>(eyes_index.at(2))+shape.at<float>(eyes_index.at(3)) )*0.5;
        float ry = ( shape.at<float>(eyes_index.at(2)+numLandmarks)+shape.at<float>(eyes_index.at(3)+numLandmarks) )*0.5;
        float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );//计算两眼的距离
        for(int j=0; j<numLandmarks; j++){
            float dx = target_shape.at<float>(i, j) - current_shape.at<float>(i, j);
            float dy = target_shape.at<float>(i, j+numLandmarks) - current_shape.at<float>(i, j+numLandmarks);
            error0 += sqrt(dx*dx + dy*dy)/distance;
        }
    }
    error0 = error0/samplesNum/numLandmarks;
    std::cout <<"初始误差为: " << error0 << "\n" << std::endl;


    for(int i=0; i<LinearRegressors.size(); i++){
        //开始计算描述子
        int bins = 1;
        for(int j=0; j<HoGParams.at(i).num_bins; j++)
            bins = 2*bins;
        cv::Mat HogDescriptors(samplesNum, (bins*HoGParams.at(i).num_cells*HoGParams.at(i).num_cells)*LandmarkIndexs.at(i).size()+1, CV_32FC1);
        for(int j=0; j<samplesNum; j++){
            cv::Mat grayImage = cv::imread(mImageLabels.at(j).imagePath, CV_LOAD_IMAGE_GRAYSCALE);
            cv::Mat Descriptor = CalculateHogDescriptor(grayImage, current_shape.row(j), LandmarkIndexs.at(i), eyes_index, HoGParams.at(i));
            assert(Descriptor.cols == HogDescriptors.cols);
            for(int k=0; k<Descriptor.cols; k++){
                HogDescriptors.at<float>(j, k) = Descriptor.at<float>(0,k);
            }
        }
        //描述子计算完成，开始一次迭代
        cv::Mat update_step = target_shape - current_shape;
        int numLandmarks = update_step.cols/2;
        if(isNormal){
            for(int j=0; j<samplesNum; j++){
                cv::Mat shape = current_shape.row(j);
                float lx = ( shape.at<float>(eyes_index.at(0))+shape.at<float>(eyes_index.at(1)) )*0.5;
                float ly = ( shape.at<float>(eyes_index.at(0)+numLandmarks)+shape.at<float>(eyes_index.at(1)+numLandmarks) )*0.5;
                float rx = ( shape.at<float>(eyes_index.at(2))+shape.at<float>(eyes_index.at(3)) )*0.5;
                float ry = ( shape.at<float>(eyes_index.at(2)+numLandmarks)+shape.at<float>(eyes_index.at(3)+numLandmarks) )*0.5;
                float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );
                update_step.row(j) = update_step.row(j)/distance;
            }
        }
        LinearRegressors.at(i).learn(HogDescriptors, update_step);
        update_step = LinearRegressors.at(i).predict(HogDescriptors);
        if(isNormal){
            for(int j=0; j<samplesNum; j++){
                cv::Mat shape = current_shape.row(j);
                float lx = ( shape.at<float>(eyes_index.at(0))+shape.at<float>(eyes_index.at(1)) )*0.5;
                float ly = ( shape.at<float>(eyes_index.at(0)+numLandmarks)+shape.at<float>(eyes_index.at(1)+numLandmarks) )*0.5;
                float rx = ( shape.at<float>(eyes_index.at(2))+shape.at<float>(eyes_index.at(3)) )*0.5;
                float ry = ( shape.at<float>(eyes_index.at(2)+numLandmarks)+shape.at<float>(eyes_index.at(3)+numLandmarks) )*0.5;
                float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );
                update_step.row(j) = update_step.row(j)*distance;
            }
        }
        current_shape = current_shape + update_step;
        //一次迭代结束，更新梯度变化，计算误差

        float error = 0.0f;
        for(int i=0; i<samplesNum; i++){
            cv::Mat shape = current_shape.row(i);
            float lx = ( shape.at<float>(eyes_index.at(0))+shape.at<float>(eyes_index.at(1)) )*0.5;
            float ly = ( shape.at<float>(eyes_index.at(0)+numLandmarks)+shape.at<float>(eyes_index.at(1)+numLandmarks) )*0.5;
            float rx = ( shape.at<float>(eyes_index.at(2))+shape.at<float>(eyes_index.at(3)) )*0.5;
            float ry = ( shape.at<float>(eyes_index.at(2)+numLandmarks)+shape.at<float>(eyes_index.at(3)+numLandmarks) )*0.5;
            float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );//计算两眼的距离
            for(int j=0; j<numLandmarks; j++){
                float dx = target_shape.at<float>(i, j) - current_shape.at<float>(i, j);
                float dy = target_shape.at<float>(i, j+numLandmarks) - current_shape.at<float>(i, j+numLandmarks);
                error += sqrt(dx*dx + dy*dy)/distance;
            }
        }
        error = error/samplesNum/numLandmarks;
        std::cout << "现在平均误差是: " << error << "\n" << std::endl;
    }
}



cv::Mat ldmarkmodel::predict(const cv::Mat& src){
    cv::Mat grayImage;
    if(src.channels() == 1){
        grayImage = src;
    }else if(src.channels() == 3){
        cv::cvtColor(src, grayImage, CV_BGR2GRAY);
    }else if(src.channels() == 4){
        cv::cvtColor(src, grayImage, CV_RGBA2GRAY);
    }else{
        return cv::Mat();
    }

    std::vector<cv::Rect> mFaceRects;
    face_cascade.detectMultiScale(grayImage, mFaceRects, 1.2, 2, 0, cv::Size(50, 50));
    if(mFaceRects.size() <=0)
        return cv::Mat();

    FACE_TRACE("mFaceRects.size %d  " , mFaceRects.size() );

    cv::Rect maxRect = mFaceRects[0];
    for(int i=1; i<mFaceRects.size(); i++){
        if(maxRect.area() < mFaceRects[i].area())
            maxRect = mFaceRects[i];
    }
    cv::Mat current_shape = align_mean(meanShape, maxRect);        //perturb(maxRect)
    int numLandmarks = current_shape.cols/2;

    cv::Mat drawImage = src.clone();
    for(int j=0; j<numLandmarks; j++){
        int x = current_shape.at<float>(j);
        int y = current_shape.at<float>(j + numLandmarks);
        cv::circle(drawImage, cv::Point(x, y), 3, cv::Scalar(255, 0, 0), -1);
    }
    cv::imshow("drawImage", drawImage);
    cv::waitKey(0);


    for(int i=0; i<LinearRegressors.size(); i++){
        cv::Mat Descriptor = CalculateHogDescriptor(grayImage, current_shape, LandmarkIndexs.at(i), eyes_index, HoGParams.at(i));
        cv::Mat update_step = LinearRegressors.at(i).predict(Descriptor);
        if(isNormal){
            float lx = ( current_shape.at<float>(eyes_index.at(0))+current_shape.at<float>(eyes_index.at(1)) )*0.5;
            float ly = ( current_shape.at<float>(eyes_index.at(0)+numLandmarks)+current_shape.at<float>(eyes_index.at(1)+numLandmarks) )*0.5;
            float rx = ( current_shape.at<float>(eyes_index.at(2))+current_shape.at<float>(eyes_index.at(3)) )*0.5;
            float ry = ( current_shape.at<float>(eyes_index.at(2)+numLandmarks)+current_shape.at<float>(eyes_index.at(3)+numLandmarks) )*0.5;
            float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );
            update_step = update_step*distance;
        }
        current_shape = current_shape + update_step;

        drawImage.release();
        drawImage = src.clone();
        for(int j=0; j<numLandmarks; j++){
            int x = current_shape.at<float>(j);
            int y = current_shape.at<float>(j + numLandmarks);
            cv::circle(drawImage, cv::Point(x, y), 3, cv::Scalar(255, 0, 0), -1);
        }
        cv::imshow("drawImage", drawImage);
        cv::waitKey(0);
    }
    return current_shape;
}


int ldmarkmodel::track(const cv::Mat& src, cv::Mat& current_shape, bool isDetFace){
    cv::Mat grayImage;
    FACE_TRACE( "src channel %d " , src.channels() );
    FACE_TRACE(" track LinearRegressors.size() %d " , LinearRegressors.size() );
    if(src.channels() == 1){
        grayImage = src;
    }else if(src.channels() == 3){
        cv::cvtColor(src, grayImage, CV_BGR2GRAY);
    }else if(src.channels() == 4){
        cv::cvtColor(src, grayImage, CV_RGBA2GRAY);
    }else{
        return SDM_ERROR_IMAGE;
    }
    FACE_TRACE( "grayImage channel %d " , grayImage.channels() );

    if(!current_shape.empty()){
        faceBox = get_enclosing_bbox(current_shape);
    }else{
        faceBox = cv::Rect(0,0,0,0);
    }
    int error_code = SDM_NO_ERROR;
    cv::Rect mfaceBox = faceBox & cv::Rect(0, 0, grayImage.cols, grayImage.rows);
    float ratio = ((float)faceBox.width)/faceBox.height;
//    if(isDetFace || faceBox.area()<10000 || ratio>1.45f || ratio<0.8f || ((float)mfaceBox.area())/faceBox.area()<0.85f){
//        std::vector<cv::Rect> mFaceRects;
//        face_cascade.detectMultiScale(grayImage, mFaceRects, 1.3, 3, 0, cv::Size(100, 100));
//        if(mFaceRects.size() <=0){
//            current_shape = cv::Mat();
//            return SDM_ERROR_FACENO;
//        }
//        faceBox = mFaceRects[0];
//        for(int i=1; i<mFaceRects.size(); i++){
//            if(faceBox.area() < mFaceRects[i].area())
//                faceBox = mFaceRects[i];
//        }
//        error_code = SDM_ERROR_FACEDET;
//    }
    FACE_TRACE("detectMultiScale start...  isDetFace %d  area %d " ,isDetFace ,  faceBox.area() );
    if(isDetFace || faceBox.area()<100){
        std::vector<cv::Rect> mFaceRects;
        face_cascade.detectMultiScale(grayImage, mFaceRects,  1.1, 2,  0 | cv::CASCADE_SCALE_IMAGE,  cv::Size(60, 60));
        if(mFaceRects.size() <=0){
            current_shape = cv::Mat();
            FACE_TRACE("no face deteced!");
            return SDM_ERROR_FACENO;
        }
        FACE_TRACE("face rects %d " , mFaceRects.size() );

		if (mFaceRects.size() >= 1) {
			FACE_TRACE("mFaceRects 0   x:%d y:%d w:%d h %d ", mFaceRects[0].x, mFaceRects[0].y, mFaceRects[0].height, mFaceRects[0].width);
		}

        FACE_TRACE("face area :%d", faceBox.area());
        FACE_TRACE("face area x:%d y:%d w:%d h %d ", faceBox.x , faceBox.y, faceBox.height , faceBox.width );
        for(int i= 0 ; i<mFaceRects.size(); i++){
			if (faceBox.area() < mFaceRects[i].area()) {
				faceBox = mFaceRects[i];
			}
                
        }// 显示最大的
        //faceBox = mFaceRects[mFaceRects.size() - 1 ];
        error_code = SDM_NO_ERROR;
    }

	FACE_TRACE("choose face area:x:%d y:%d w:%d h %d ", faceBox.x, faceBox.y, faceBox.height, faceBox.width);
    // faceBox 把人脸框出来


    current_shape = align_mean(meanShape, faceBox); // meanShape  dims 2  rows 1 cols 136 
	// 返回的current_shape也是 dims 2 rows 1 cols 136
	// 根据平均脸部特征点 计算出的给点脸框 的 特征点位置  


    int numLandmarks = current_shape.cols/2; // 136 / 2 = 68 
    FACE_TRACE("LinearRegressors.size() %d " , LinearRegressors.size() );

    for(int i=0; i<LinearRegressors.size(); i++){
        //FACE_TRACE("face CalculateHogDescriptor:%d", i);
        cv::Mat Descriptor = CalculateHogDescriptor(grayImage, current_shape, LandmarkIndexs.at(i), eyes_index, HoGParams.at(i));
        //FACE_TRACE("face predict:%d", i);
        cv::Mat update_step = LinearRegressors.at(i).predict(Descriptor);
        //FACE_TRACE("face update_step:%d", i);
        if(isNormal){ // true 
            float lx = (   current_shape.at<float>(eyes_index.at(0))                +   current_shape.at<float>(eyes_index.at(1))                )*0.5;
            float ly = (   current_shape.at<float>(eyes_index.at(0)+numLandmarks)   +   current_shape.at<float>(eyes_index.at(1)+numLandmarks)   )*0.5;
            float rx = (   current_shape.at<float>(eyes_index.at(2))                +   current_shape.at<float>(eyes_index.at(3))                )*0.5;
            float ry = (   current_shape.at<float>(eyes_index.at(2)+numLandmarks)   +   current_shape.at<float>(eyes_index.at(3)+numLandmarks)   )*0.5;
            float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );
            update_step =  update_step * distance;
        }
        current_shape = current_shape + update_step;
    }
    return error_code;
}

void ldmarkmodel::printmodel(){
    if(isNormal)
        std::cout << "以两眼距离归一化步长" << std::endl;
    else
        std::cout << "不归一化步长" << std::endl;
    std::cout << "一共" << LinearRegressors.size() << "次迭代回归..." << std::endl;
    for(int i=0; i<LandmarkIndexs.size(); i++){
        std::cout <<"第"<<i<<"次回归: "<<LandmarkIndexs.at(i).size()<<"个点  ";
        std::cout << "num_cells:"<<HoGParams.at(i).num_cells<<"  cell_size:"<<HoGParams.at(i).cell_size <<"  num_bins:"<<HoGParams.at(i).num_bins<<"  relative_patch_size:"<<HoGParams.at(i).relative_patch_size<<std::endl;
    }
}

void ldmarkmodel::convert(std::vector<int> &full_eyes_Indexs){
    std::vector<int> tar_LandmarkIndex;
    for(int i=0; i<LinearRegressors.size(); i++){
        for(int j=0; j<LandmarkIndexs.at(i).size(); j++){
            int t = LandmarkIndexs.at(i).at(j);
            bool flag = true;
            for(int k=0; k<tar_LandmarkIndex.size(); k++){
                if(t == tar_LandmarkIndex.at(k)){
                    flag = false;
                    break;
                }
            }
            if(flag)
                tar_LandmarkIndex.push_back(t);
        }
    }
    for(int i=0; i<full_eyes_Indexs.size(); i++){
        int t = full_eyes_Indexs.at(i);
        bool flag = true;
        for(int j=0; j<tar_LandmarkIndex.size(); j++){
            if(t == tar_LandmarkIndex.at(j)){
                flag = false;
                break;
            }
        }
        if(flag)
            tar_LandmarkIndex.push_back(t);
    }
    //更新转换meanShape
    cv::Mat tmp = meanShape.clone();
    meanShape.release();
    meanShape.create(1, tar_LandmarkIndex.size()*2, tmp.type());
    for(int i=0; i<tar_LandmarkIndex.size(); i++){
        meanShape.at<float>(i) = tmp.at<float>(tar_LandmarkIndex.at(i));
        meanShape.at<float>(i+tar_LandmarkIndex.size()) = tmp.at<float>(tar_LandmarkIndex.at(i)+tmp.cols/2);
    }
    //更新转换LandmarkIndexs
    for(int i=0; i<LinearRegressors.size(); i++){
        for(int j=0; j<LandmarkIndexs.at(i).size(); j++){
            for(int k=0; k<tar_LandmarkIndex.size(); k++){
                if(LandmarkIndexs.at(i).at(j) == tar_LandmarkIndex.at(k)){
                    LandmarkIndexs.at(i).at(j) = k;
                    break;
                }
            }
        }
    }
    //更新转换eyes_index
    for(int i=0; i<eyes_index.size(); i++){
        bool flag = false;
        for(int j=0; i<tar_LandmarkIndex.size(); j++){
            if(eyes_index.at(i) == tar_LandmarkIndex.at(j)){
                eyes_index.at(i) = j;
                flag = true;
                break;
            }
        }
        assert(flag);
    }
    //更新转换LinearRegressors
    for(int i=0; i<LinearRegressors.size(); i++){
        LinearRegressors.at(i).convert(tar_LandmarkIndex);
    }
}

void ldmarkmodel::EstimateHeadPose(cv::Mat &current_shape, cv::Vec3d &eav){
    if(current_shape.empty())
        return;
    static const int samplePdim = 7;
    float miny = 10000000000.0f;
    float maxy = 0.0f;
    float sumx = 0.0f;
    float sumy = 0.0f;
    for(int i=0; i<samplePdim; i++){
        sumx += current_shape.at<float>(estimateHeadPosePointIndexs[i]);
        float y = current_shape.at<float>(estimateHeadPosePointIndexs[i]+current_shape.cols/2);
        sumy += y;
        if(miny > y)
            miny = y;
        if(maxy < y)
            maxy = y;
    }
    float dist = maxy - miny;
    sumx = sumx/samplePdim;
    sumy = sumy/samplePdim;
    static cv::Mat tmp(1, 2*samplePdim+1, CV_32FC1);
    for(int i=0; i<samplePdim; i++){
        tmp.at<float>(i) = (current_shape.at<float>(estimateHeadPosePointIndexs[i])-sumx)/dist;
        tmp.at<float>(i+samplePdim) = (current_shape.at<float>(estimateHeadPosePointIndexs[i]+current_shape.cols/2)-sumy)/dist;
    }
    tmp.at<float>(2*samplePdim) = 1.0f;
//    cv::Mat predict = tmp*estimateHeadPoseMat;
//    double _pm[12] = {predict.at<float>(0), predict.at<float>(1), predict.at<float>(2), 0,
//                      predict.at<float>(3), predict.at<float>(4), predict.at<float>(5), 0,
//                      predict.at<float>(6), predict.at<float>(7), predict.at<float>(8), 0};
//    cv::Mat tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
//    cv::decomposeProjectionMatrix(cv::Mat(3,4,CV_64FC1,_pm),tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,eav);
    cv::Mat predict = tmp*estimateHeadPoseMat2;
    eav[0] = predict.at<float>(0);
    eav[1] = predict.at<float>(1);
    eav[2] = predict.at<float>(2);
//    std::cout << eav[0] << "  " << eav[1] << "  " << eav[2] << std::endl;
//    Pitch = eav[0];
//    Yaw	= eav[1];
//    Roll  = eav[2];
    return;
}

cv::Mat ldmarkmodel::EstimateHeadPose(cv::Mat &current_shape){
    if(current_shape.empty())
        return cv::Mat();
    static const int samplePdim = 7;
    float miny = 10000000000.0f;
    float maxy = 0.0f;
    float sumx = 0.0f;
    float sumy = 0.0f;
    for(int i=0; i<samplePdim; i++){
        sumx += current_shape.at<float>(estimateHeadPosePointIndexs[i]);
        float y = current_shape.at<float>(estimateHeadPosePointIndexs[i]+current_shape.cols/2);
        sumy += y;
        if(miny > y)
            miny = y;
        if(maxy < y)
            maxy = y;
    }
    float dist = maxy - miny;
    sumx = sumx/samplePdim;
    sumy = sumy/samplePdim;
    static cv::Mat tmp(1, 2*samplePdim+1, CV_32FC1);
    for(int i=0; i<samplePdim; i++){
        tmp.at<float>(i) = (current_shape.at<float>(estimateHeadPosePointIndexs[i])-sumx)/dist;
        tmp.at<float>(i+samplePdim) = (current_shape.at<float>(estimateHeadPosePointIndexs[i]+current_shape.cols/2)-sumy)/dist;
    }
    tmp.at<float>(2*samplePdim) = 1.0f;
    cv::Mat predict = tmp*estimateHeadPoseMat;
    cv::Mat rot(3,3,CV_32FC1);
    for(int i=0; i<3; i++){
        rot.at<float>(i,0) = predict.at<float>(3*i);
        rot.at<float>(i,1) = predict.at<float>(3*i+1);
        rot.at<float>(i,2) = predict.at<float>(3*i+2);
    }
    return rot;
}

void ldmarkmodel::drawPose(cv::Mat& img, const cv::Mat& current_shape, float lineL)
{
    if(current_shape.empty())
        return;
    static const int samplePdim = 7;
    float miny = 10000000000.0f;
    float maxy = 0.0f;
    float sumx = 0.0f;
    float sumy = 0.0f;
    for(int i=0; i<samplePdim; i++){
        sumx += current_shape.at<float>(estimateHeadPosePointIndexs[i]);
        float y = current_shape.at<float>(estimateHeadPosePointIndexs[i]+current_shape.cols/2);
        sumy += y;
        if(miny > y)
            miny = y;
        if(maxy < y)
            maxy = y;
    }
    float dist = maxy - miny;
    sumx = sumx/samplePdim;
    sumy = sumy/samplePdim;
    static cv::Mat tmp(1, 2*samplePdim+1, CV_32FC1);
    for(int i=0; i<samplePdim; i++){
        tmp.at<float>(i) = (current_shape.at<float>(estimateHeadPosePointIndexs[i])-sumx)/dist;
        tmp.at<float>(i+samplePdim) = (current_shape.at<float>(estimateHeadPosePointIndexs[i]+current_shape.cols/2)-sumy)/dist;
    }
    tmp.at<float>(2*samplePdim) = 1.0f;
    cv::Mat predict = tmp*estimateHeadPoseMat;
    cv::Mat rot(3,3,CV_32FC1);
    for(int i=0; i<3; i++){
        rot.at<float>(i,0) = predict.at<float>(3*i);
        rot.at<float>(i,1) = predict.at<float>(3*i+1);
        rot.at<float>(i,2) = predict.at<float>(3*i+2);
    }
    //we have get the rot mat
    int loc[2] = {70, 70};
    int thickness = 2;
    int lineType  = 8;

    cv::Mat P = (cv::Mat_<float>(3,4) <<
        0, lineL, 0,  0,
        0, 0, -lineL, 0,
        0, 0, 0, -lineL);
    P = rot.rowRange(0,2)*P;
    P.row(0) += loc[0];
    P.row(1) += loc[1];
    cv::Point p0(P.at<float>(0,0),P.at<float>(1,0));

    line(img, p0, cv::Point(P.at<float>(0,1),P.at<float>(1,1)), cv::Scalar( 255, 0, 0 ), thickness, lineType);
    line(img, p0, cv::Point(P.at<float>(0,2),P.at<float>(1,2)), cv::Scalar( 0, 255, 0 ), thickness, lineType);
    line(img, p0, cv::Point(P.at<float>(0,3),P.at<float>(1,3)), cv::Scalar( 0, 0, 255 ), thickness, lineType);

    //printf("%f %f %f\n", rot.at<float>(0, 0), rot.at<float>(0, 1), rot.at<float>(0, 2));
    //printf("%f %f %f\n", rot.at<float>(1, 0), rot.at<float>(1, 1), rot.at<float>(1, 2));

    cv::Vec3d eav;
    cv::Mat tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
    double _pm[12] = {rot.at<float>(0, 0), rot.at<float>(0, 1),rot.at<float>(0, 2), 0,
                      rot.at<float>(1, 0), rot.at<float>(1, 1),rot.at<float>(1, 2),0,
                      rot.at<float>(2, 0), rot.at<float>(2, 1),rot.at<float>(2, 2),0};
    cv::decomposeProjectionMatrix(cv::Mat(3,4,CV_64FC1,_pm),tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,eav);
    std::stringstream ss;
    ss << eav[0];
    std::string txt = "Pitch: " + ss.str();
    cv::putText(img, txt,  cv::Point(60, 20), 0.5,0.5, cv::Scalar(0,0,255));
    std::stringstream ss1;
    ss1 << eav[1];
    std::string txt1 = "Yaw: " + ss1.str();
    cv::putText(img, txt1, cv::Point(60, 40), 0.5,0.5, cv::Scalar(0,0,255));
    std::stringstream ss2;
    ss2 << eav[2];
    std::string txt2 = "Roll: " + ss2.str();
    cv::putText(img, txt2, cv::Point(60, 60), 0.5,0.5, cv::Scalar(0,0,255));

    predict = tmp*estimateHeadPoseMat2;
    std::stringstream ss3;
    ss3 << predict.at<float>(0);
    txt = "Pitch: " + ss3.str();
    cv::putText(img, txt,  cv::Point(340, 20), 0.5,0.5, cv::Scalar(255,255,255));
    std::stringstream ss4;
    ss4 << predict.at<float>(1);
    txt1 = "Yaw: " + ss4.str();
    cv::putText(img, txt1, cv::Point(340, 40), 0.5,0.5, cv::Scalar(255,255,255));
    std::stringstream ss5;
    ss5 << predict.at<float>(2);
    txt2 = "Roll: " + ss5.str();
    cv::putText(img, txt2, cv::Point(340, 60), 0.5,0.5, cv::Scalar(255,255,255));
//        Pitch = eav[0];
//        Yaw	  = eav[1];
//        Roll  = eav[2];
}
//加载模型
bool load_ldmarkmodel(std::string filename, ldmarkmodel &model)
{
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open())
        return false;
    cereal::BinaryInputArchive input_archive(file);
    input_archive(model);
    file.close();
    return true;
}

//保存模型
void save_ldmarkmodel(ldmarkmodel model, std::string filename)
{
    std::ofstream file(filename, std::ios::binary);
    cereal::BinaryOutputArchive output_archive(file);
    output_archive(model);
    file.close();
}


