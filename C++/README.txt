文件清单
liveness.cpp：活体检测程序
liveness.h：活体检测头文件
svm_model：训练出来的svm模型
svm_minmax：测试时特征归一化的参数
./lbp：LBP特征提取库
./libsvm-3.19：LibSVM库

需要加入项目的cpp文件：
liveness.cpp
./lbp/lbp.cpp
./lbp/histogram.cpp
./libsvm-3.19/svm.cpp
引用的头文件为：
#include"liveness.h"

使用的OpenCV版本为3.0.0-RC1

使用的人脸数据库包括NUAA, PRINT-ATTACK, CASIA
这些库不包括在最终程序中

活体检测的类如下
class ThuVisionFaceLiveCheck{
public:
	// Initialization: successful - true; failed - false
	bool Init();
	// Liveness Predict: input image path or Mat. 
	// If it is a real face, return true; otherwise, return false
	bool Check(std::string imgPath);
	bool Check(cv::Mat img);
private:
	int img_size;
	int LBP_size;
	svm_model* model;
	double* Feature_Max;
	double* Feature_Min;
	vector<int> mapping_r8;
	vector<int> mapping_r16;
};