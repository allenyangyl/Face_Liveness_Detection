#include"liveness.h"

// Demo of Liveness Detection
int main()
{
	Test();
	/*
	// Initialization
	ThuVisionFaceLiveCheck Check;
	bool Init_message = Check.Init();
	if(Init_message == false)
	{
		std::cout << "Initialization failed. " << endl;
		return -1;
	}
	std::cout << "Initialized successfully. " << endl;

	// Check with Image Path
	string imgPath0 = "00000.png";
	double value0 = Check.Check(imgPath0);
	cout << imgPath0 << " is a " << (value0 >= 0.5 ? "real" : "fake") << " face. " <<endl;

	// Check with Image Mat
	string imgPath1 = "00001.png";
	Mat img = imread(imgPath1);
	if(img.empty())
	{
		std::cout << "Can not load image: " << imgPath1 << endl;
		return -1;
	}
	double value1 = Check.Check(img);
	cout << "img" << " is a " << (value1 >= 0.5 ? "real" : "fake") << " face. " <<endl;
	*/

	return 0;
}
