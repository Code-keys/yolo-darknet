#pragma once
#include "myutils.h"
class CReadImage
{
public:
	CReadImage();
	~CReadImage();
public:
	bool set_fname(const char* s);
	void set_params(float coef) { m_coef = coef; }
	void set_image(cv::Mat& im);
	Mat& get_image();
private:
	bool process_img();
	bool compute_thresh(Mat& in_im, float coef, double& minV, double& maxV);
	bool convert_tif(Mat& in_im, Mat& out, double minV, double maxV);
	bool release_image();
private:
	char fname[256];
	float m_coef;
	Mat in_im, out_im;
};

