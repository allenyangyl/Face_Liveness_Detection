#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <opencv2\opencv.hpp>
#include ".\libsvm-3.19\svm.h" 
#include ".\lbp\lbp.hpp" 
#include ".\lbp\histogram.hpp" 
using namespace std;
using namespace cv;

// Sigmoid
double sigmoid(double input);

// DoG Features
Mat DoG(Mat input, double sigma1 = 0.5, double sigma2 = 1);

// LBP Features
Mat LBP(Mat input, vector<int> mapping_r8, vector<int> mapping_r16);

// Extract Features
svm_node* Liveness_Feature(Mat img, vector<int> mapping_r8, vector<int> mapping_r16, int img_size, int LBP_size);

// Predict
double Liveness_Predict(Mat img, svm_model* model, double* Feature_Max, double* Feature_Min, 
						vector<int> mapping_r8, vector<int> mapping_r16, int img_size, int LBP_size);

// Liveness Detection Class
class ThuVisionFaceLiveCheck{
public:
	// Initialization: successful - true; failed - false
	bool Init();
	// Liveness Predict: input image path or Mat. 
	double Check(std::string imgPath);
	double Check(cv::Mat img);
private:
	// Image size (DoG feature size = img_size * img_size)
	int img_size;
	// LBP feature size
	int LBP_size;
	// SVM Model
	svm_model* model;
	// Min and Max (for scaling)
	double* Feature_Max;
	double* Feature_Min;
	// LBP mapping
	vector<int> mapping_r8;
	vector<int> mapping_r16;
};

// Training
void Train();

// Testing
void Test();
