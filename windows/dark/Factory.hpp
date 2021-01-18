#pragma once

#include <iostream>
#include <math.h>
#include <string>
#include <algorithm>
//#include <opencv2/highgui.hpp>
//#include <opencv2/opencv.hpp>

namespace images {
	int interface();
	int interface1();

	struct person {
		std::string names;
		int ages;
		person(std::string x, int y) :names(x), ages(y) {};
	};

}