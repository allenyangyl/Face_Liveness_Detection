#include"liveness.h"
	
// Sigmoid
double sigmoid(double input)
{
	return 1 / ( 1 + exp(-input) );
}

// DoG Features
Mat DoG(Mat input, double sigma1, double sigma2)
{
	Mat XF1, XF2, DXF, output;
	int size1, size2;
	// Filter Sizes
	size1 = 2 * (int)(3*sigma1) + 3;
	size2 = 2 * (int)(3*sigma2) + 3;
	// Gaussian Filter
	GaussianBlur(input, XF1, Size(size1, size1), sigma1, sigma1, BORDER_REPLICATE);
	GaussianBlur(input, XF2, Size(size2, size2), sigma2, sigma2, BORDER_REPLICATE);
	// Difference
	DXF = XF1 - XF2;
	// Discrete Fourier Transform
	DXF.convertTo(DXF, CV_64FC1);
	dft(DXF, output);
	return abs(output);
}

// LBP Features
Mat LBP(Mat input, vector<int> mapping_r8, vector<int> mapping_r16)
{

	// LBP(r2, n16)
	Mat LBP_16_2 = lbp::ELBP(input, 2, 16);
	for(int i = 0; i < LBP_16_2.rows; i++)
		for(int j = 0; j < LBP_16_2.cols; j++)
			LBP_16_2.at<int>(i, j) = mapping_r16[LBP_16_2.at<int>(i, j)];
	Mat hist_16_2 = lbp::histogram(LBP_16_2, 243);

	// LBP(r2, n8)
	Mat LBP_8_2 = lbp::ELBP(input, 2, 8);
	for(int i = 0; i < LBP_8_2.rows; i++)
		for(int j = 0; j < LBP_8_2.cols; j++)
			LBP_8_2.at<int>(i, j) = mapping_r8[LBP_8_2.at<int>(i, j)];
	Mat hist_8_2 = lbp::histogram(LBP_8_2, 59);

	// LBP(r1, n8) * 9
	Mat LBP_8_1 = lbp::ELBP(input, 1, 8);
	for(int i = 0; i < LBP_8_1.rows; i++)
		for(int j = 0; j < LBP_8_1.cols; j++)
			LBP_8_1.at<int>(i, j) = mapping_r8[LBP_8_1.at<int>(i, j)];
	Mat hist_8_1 = lbp::spatial_histogram(LBP_8_1, 59, Size_<int>(30, 30), 14);

	// Histogram Concatenate
	Mat hist_8, hist;
	hconcat(hist_8_2, hist_8_1, hist_8);
	hconcat(hist_16_2, hist_8, hist);
	return hist;
}

// Extract Features
svm_node* Liveness_Feature(Mat img, vector<int> mapping_r8, vector<int> mapping_r16, int img_size, int LBP_size)
{
	// Total feature size
	int fea_size = img_size * img_size + LBP_size;
	// Grayscale
	cvtColor(img, img, CV_BGR2GRAY);
	// Histogram Equlization - Ineffective
	//equalizeHist(img, img);
	// Image Resize
	resize(img, img, Size(img_size, img_size));
	// DoG Features
	Mat DoG_Feature = DoG(img);
	// LBP Features
	Mat LBP_Feature = LBP(img, mapping_r8, mapping_r16);
	// Filling Features in svm_node
	svm_node* features = new svm_node[fea_size+1];
	for(int i = 0; i < DoG_Feature.rows; i++)
		for(int j = 0; j < DoG_Feature.cols; j++)
		{
			features[i*img_size+j].index = i*img_size+j+1;
			features[i*img_size+j].value = DoG_Feature.at<double>(i, j);
		}
	for(int i = 0; i < LBP_Feature.cols; i++)
		{
			features[img_size*img_size+i].index = img_size*img_size+i+1;
			features[img_size*img_size+i].value = (double)LBP_Feature.at<int>(0, i);
		}
	features[fea_size].index = -1;
	// Return
	return features;
}

// Prediction
double Liveness_Predict(Mat img, svm_model* model, double* Feature_Max, double* Feature_Min, 
						vector<int> mapping_r8, vector<int> mapping_r16, int img_size, int LBP_size)
{
	int fea_size = img_size * img_size + LBP_size;
	svm_node* features = Liveness_Feature(img, mapping_r8, mapping_r16, img_size, LBP_size);
	for(int i = 0; i < fea_size; i++)
		features[i].value = (features[i].value - Feature_Min[i]) / (Feature_Max[i] - Feature_Min[i]);
	double Value;
	double Label = svm_predict_values(model, features, &Value);
	delete[] features;
	return Value;
}

// Initialization
bool ThuVisionFaceLiveCheck::Init()
{
	// Image size, LBP feature size and total feature size
	img_size = 64;
	LBP_size = 243 + 59 + 9*59;
	int fea_size = img_size * img_size + LBP_size;

	// Path
	char* svm_path = "svm_model";
	char* minmax_path = "svm_minmax.txt";
	string Mapping_r8_path = ".\\lbp\\mapping_n8.txt";
	string Mapping_r16_path = ".\\lbp\\mapping_n16.txt";

	// Load Model
	svm_model* SVM_Model = svm_load_model(svm_path);
	if(model == NULL)
	{
		std::cout << "Can not open file: " << svm_path << endl;
		return false;
	}
	model = SVM_Model;

	// Scaling
	ifstream feature_minmax_file;
	string feature_minmax_path = minmax_path;
	feature_minmax_file.open(feature_minmax_path);
	if(!feature_minmax_file.is_open())
	{
		std::cout << "Can not open file: " << feature_minmax_path << endl;
		return false;
	}
	double* SVM_Feature_Max = new double[fea_size];
	double* SVM_Feature_Min = new double[fea_size];
	for(int Feature_Idx = 0; Feature_Idx < fea_size; Feature_Idx++)
		feature_minmax_file >> SVM_Feature_Max[Feature_Idx] >> SVM_Feature_Min[Feature_Idx];
	Feature_Max = SVM_Feature_Max;
	Feature_Min = SVM_Feature_Min;
	
	int mapping;
	// LBP n=8 mapping
	vector<int> SVM_mapping_r8;
	ifstream Mapping_r8_file;
	Mapping_r8_file.open(Mapping_r8_path);
	if(!Mapping_r8_file.is_open())
	{
		std::cout << "Can not open file: " << Mapping_r8_path << endl;
		return false;
	}
	for(int i = 0; i < 256; i++)
	{
		Mapping_r8_file >> mapping;
		SVM_mapping_r8.push_back(mapping);
	}
	mapping_r8 = SVM_mapping_r8;

	// LBP n=16 mapping
	vector<int> SVM_mapping_r16;
	ifstream Mapping_r16_file;
	Mapping_r16_file.open(Mapping_r16_path);
	if(!Mapping_r16_file.is_open())
	{
		std::cout << "Can not open file: " << Mapping_r16_path << endl;
		return false;
	}
	for(int i = 0; i < 65536; i++)
	{
		Mapping_r16_file >> mapping;
		SVM_mapping_r16.push_back(mapping);
	}
	mapping_r16 = SVM_mapping_r16;

	return true;
}

// Check with image as an input
double ThuVisionFaceLiveCheck::Check(cv::Mat img)
{
	// Predict
	double value = Liveness_Predict(img, model, Feature_Max, Feature_Min, mapping_r8, mapping_r16, img_size, LBP_size);
	return sigmoid(value);
}

// Check with path as an input
double ThuVisionFaceLiveCheck::Check(std::string imgPath)
{
	// Load Image
	Mat img = imread(imgPath);
	if(img.empty())
	{
		std::cout << "Can not load image: " << imgPath << endl;
		return false;
	}

	return ThuVisionFaceLiveCheck::Check(img);
}

void Train()
{
	// Image size, LBP feature size and total feature size
	int img_size = 64;
	int LBP_size = 243 + 59 + 9*59;
	int fea_size = img_size * img_size + LBP_size;

	// Path
	char* svm_path = "svm_model";
	char* minmax_path = "svm_minmax.txt";
	string Mapping_r8_path = ".\\lbp\\mapping_n8.txt";
	string Mapping_r16_path = ".\\lbp\\mapping_n16.txt";

	// Train data set
	int real_train_count[] = {1742, 22274, 6322};
	int fake_train_count[] = {1748, 14064, 22080};
	string train_set[] = {"NUAA", "PRINT-ATTACK", "CASIA"};
	string real_train_name;
	string fake_train_name;
	vector<double> real_train_value;
	vector<double> fake_train_value;

	// LibSVM
	svm_problem svm_prob;
	svm_parameter svm_param;
	svm_prob.l = real_train_count[0] + real_train_count[1] + real_train_count[2] + 
		fake_train_count[0] + fake_train_count[1] + fake_train_count[2];
	svm_prob.y = new double[svm_prob.l];
	svm_prob.x = new svm_node*[svm_prob.l];
	svm_param.svm_type = C_SVC;
	svm_param.kernel_type = LINEAR;
	//svm_param.degree = 3;
	//svm_param.gamma = 0;
	//svm_param.coef0 = 0;
	svm_param.cache_size = 1;
	svm_param.eps = 1e-3;
	svm_param.C = 1;
	svm_param.shrinking = 1;
	svm_param.probability = 1;
	svm_param.nr_weight = NULL;
	svm_param.weight_label = NULL;
	svm_param.weight = NULL;
	
	// LBP n=8 mapping
	int mapping;
	vector<int> mapping_r8;
	ifstream Mapping_r8_file;
	Mapping_r8_file.open(Mapping_r8_path);
	for(int i = 0; i < 256; i++)
	{
		Mapping_r8_file >> mapping;
		mapping_r8.push_back(mapping);
	}
	// LBP n=16 mapping
	vector<int> mapping_r16;
	ifstream Mapping_r16_file;
	Mapping_r16_file.open(Mapping_r16_path);
	for(int i = 0; i < 65536; i++)
	{
		Mapping_r16_file >> mapping;
		mapping_r16.push_back(mapping);
	}

	// Extract Features
	int count = 0;
	for(int setIdx = 0; setIdx < 3; setIdx++)
	{
		// Real face training
		for(int imgIdx = 0; imgIdx < real_train_count[setIdx]; imgIdx++)
		{
			// Image Path
			stringstream real_train_stream;
			real_train_stream  << setw(5) << setfill('0') << imgIdx;
			real_train_name = ".\\" + train_set[setIdx] + "\\train\\real\\" + real_train_stream.str() + ".png";
			// Load Image
			Mat img = imread(real_train_name);
			if(img.empty())
			{
				std::cout << "Can not load image: " << real_train_name << endl;
				return;
			}
			// Extract Features
			svm_prob.y[count] = 1;
			svm_prob.x[count] = Liveness_Feature(img, mapping_r8, mapping_r16, img_size, LBP_size);
			std::cout << "Training data: " << train_set[setIdx] << " Real " << imgIdx + 1 << endl;
			count++;
		}

		// Fake face training
		for(int imgIdx = 0; imgIdx < fake_train_count[setIdx]; imgIdx++)
		{
			// Image Path
			stringstream fake_train_stream;
			fake_train_stream  << setw(5) << setfill('0') << imgIdx;
			fake_train_name = ".\\" + train_set[setIdx] + "\\train\\fake\\" + fake_train_stream.str() + ".png";
			// Load Image
			Mat img = imread(fake_train_name);
			if(img.empty())
			{
				std::cout << "Can not load image: " << fake_train_name << endl;
				return;
			}
			// Extract Features
			svm_prob.y[count] = -1;
			svm_prob.x[count] = Liveness_Feature(img, mapping_r8, mapping_r16, img_size, LBP_size);
			std::cout << "Training data: " << train_set[setIdx] << " Fake " << imgIdx + 1 << endl;
			count++;
		}
	}

	// Scaling
	ofstream minmax_file;
	minmax_file.open(minmax_path);
	for(int Feature_Idx = 0; Feature_Idx < fea_size; Feature_Idx++)
	{
		double Max = 0;
		double Min = 4294967296;
		// Get Max and Min
		for(int train_Idx = 0; train_Idx < svm_prob.l; train_Idx++)
		{
			Max = Max > svm_prob.x[train_Idx][Feature_Idx].value
				? Max : svm_prob.x[train_Idx][Feature_Idx].value;
			Min = Min < svm_prob.x[train_Idx][Feature_Idx].value
				? Min : svm_prob.x[train_Idx][Feature_Idx].value;
		}
		// Scaling Features
		for(int svm_train_Idx = 0; svm_train_Idx < svm_prob.l; svm_train_Idx++)
		{
			svm_prob.x[svm_train_Idx][Feature_Idx].value = 
				(svm_prob.x[svm_train_Idx][Feature_Idx].value - Min) / (Max - Min);
		}
		// Save Max and Min
		minmax_file << Max << ' ' << Min << endl;
	}

	// Training
	svm_model* svm_model = svm_train(&svm_prob, &svm_param);
	// Count correct rate
	int test_count = 0;
	for(int test_Idx = 0; test_Idx < svm_prob.l; test_Idx++)
	{
		double value = svm_predict(svm_model, svm_prob.x[test_Idx]);
		test_count += (value == svm_prob.y[test_Idx]);
	}
	cout << "Correct rate: " << (double) test_count / (svm_prob.l) << endl;
	// Free
	svm_save_model(svm_path, svm_model);
	svm_free_and_destroy_model(&svm_model);
	delete[] svm_prob.x;
	delete[] svm_prob.y;
	svm_destroy_param(&svm_param);

	return;
}

void Test()
{
	ThuVisionFaceLiveCheck liveness_test;
	bool Init_message = liveness_test.Init();
	if(Init_message == false)
	{
		std::cout << "Initialization failed. " << endl;
		return;
	}
	std::cout << "Initialized successfully. " << endl;

	// Test data set
	int real_test_count[] = {3362, 29562, 9966};
	int fake_test_count[] = {5761, 18636, 31683};
	string test_set[] = {"NUAA", "PRINT-ATTACK", "CASIA"};
	string real_test_name;
	string fake_test_name;
	vector<double> real_test_value[3];
	vector<double> fake_test_value[3];

	// Testing
	for(int setIdx = 0; setIdx < 3; setIdx++)
	{
		// Real face testing
		for(int imgIdx = 0; imgIdx < real_test_count[setIdx]; imgIdx++)
		{
			// Image Path
			stringstream real_test_stream;
			real_test_stream  << setw(5) << setfill('0') << imgIdx;
			real_test_name = ".\\" + test_set[setIdx] + "\\test\\real\\" + real_test_stream.str() + ".png";
			// Predict
			double value = liveness_test.Check(real_test_name);
			real_test_value[setIdx].push_back(value);
			std::cout << "Testing data: " << test_set[setIdx] << " Real " << imgIdx + 1 << "; "
				<< "Predict Value: " << value << "; "
				<< "True Label: 1. " << endl;
		}

		// Fake face testing
		for(int imgIdx = 0; imgIdx < fake_test_count[setIdx]; imgIdx++)
		{
			// Image Path
			stringstream fake_test_stream;
			fake_test_stream  << setw(5) << setfill('0') << imgIdx;
			fake_test_name = ".\\" + test_set[setIdx] + "\\test\\fake\\" + fake_test_stream.str() + ".png";
			// Predict
			double value = liveness_test.Check(fake_test_name);
			fake_test_value[setIdx].push_back(value);
			std::cout << "Testing data: " << test_set[setIdx] << " Fake " << imgIdx + 1 << "; "
				<< "Predict Value: " << value << "; "
				<< "True Label: 0. " << endl;
		}
	}

	// Result
	ofstream count_file[3];
	int correct_count;
	int* test_count = new int[3];
	for(int setIdx = 0; setIdx < 3; setIdx++)
	{
		count_file[setIdx].open("count_" + test_set[setIdx] + ".txt");
		test_count[setIdx] = real_test_count[setIdx] + fake_test_count[setIdx];
		for(double k = 0; k <= 1; k+=0.01)
		{
			correct_count = 0;
			for(int i = 0; i < real_test_value[setIdx].size(); i++)
				correct_count += (real_test_value[setIdx][i] >= k);
			for(int i = 0; i < fake_test_value[setIdx].size(); i++)
				correct_count += (fake_test_value[setIdx][i] < k );
			count_file[setIdx] << "k on " << test_set[setIdx] << ": " << k << endl;
			count_file[setIdx] << "Total data on " << test_set[setIdx] << ": " << test_count[setIdx] << endl;
			count_file[setIdx] << "Correct data on " << test_set[setIdx] << ": "  << correct_count << endl;
			count_file[setIdx] << "Correct rate on " << test_set[setIdx] << ": "  << (double) correct_count / test_count[setIdx] << endl;
		}
	}

	return;
}