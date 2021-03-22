#include "ReadImage.h"
//https://blog.csdn.net/daoqinglin/article/details/23628125

CReadImage::CReadImage()
{
	m_coef = 0.01;
}

CReadImage::~CReadImage()
{
	release_image();
}

bool CReadImage::release_image()
{
	if (!in_im.empty())
		in_im.release();
	if (!out_im.empty())
		out_im.release();
	return true;
}

bool CReadImage::set_fname(const char* s)
{
	sprintf(fname, "%s", s);
	release_image();
	try
	{
		Mat im = imread(fname, IMREAD_UNCHANGED);
		im.copyTo(in_im);
		return true;
	}
	catch (...)
	{
		return false;
	}
}

void CReadImage::set_image(Mat& im)
{
	if (in_im.channels() == 3)
	{
		if (!in_im.empty())
			in_im.release();
		cvtColor(im, in_im, COLOR_BGR2GRAY);
	}
	else
		in_im = im;
}

Mat& CReadImage::get_image()
{
	process_img();
	return out_im;
}

bool CReadImage::compute_thresh(Mat& in_im, float coef, double& minV, double& maxV)
{
	double minVal, maxVal;
	int    minIdx[2] = {}, maxIdx[2] = {};	// minimum Index, maximum Index
	minMaxIdx(in_im, &minVal, &maxVal, minIdx, maxIdx);

	int histBinNum = maxVal + 1;
	float range[] = { 0, maxVal};
	const float* histRange = { range };

	bool uniform = true;
	bool accumulate = false;

	Mat hist;
	calcHist(&in_im, 1, 0, Mat(), hist, 1, &histBinNum, &histRange, uniform, accumulate);
	for (int i = 1; i <= maxVal; i++)
	{
		double v = getdatafromMat(hist, i, 0, 0) + getdatafromMat(hist, i - 1, 0, 0);
		setdatatoMat(hist, i, 0, v, 0);
	}

	normalize(hist, hist, 1, 0, NORM_INF, -1, Mat());

	minV = 0;
	for (int i = 0; i <= maxVal; i++)
	{
		double v = getdatafromMat(hist, i, 0, 0);
		if (v > coef)
		{
			minV = i;
			break;
		}
	}
	maxV = maxVal;
	for (int i = 0; i <= maxVal; i++)
	{
		double v = getdatafromMat(hist, int(maxVal)-i, 0, 0);
		if (v < 1 - coef)
		{
			maxV = maxVal - i;
			break;
		}
	}
	return true;
}

bool CReadImage::convert_tif(Mat& in_im, Mat& out, double minV, double maxV)
{
	out = Mat(in_im.rows, in_im.cols, CV_8U);
	double inV, outV;
	for (int i=0; i < in_im.rows; i++)
	{
		for (int j=0; j < in_im.cols; j ++)
		{
			inV = getdatafromMat(in_im, i, j, 1);
			if (inV < minV)
				outV = inV*20.0 / minV;
			else if (inV > maxV)
				outV = 255;
			else
			{
				outV = pow((inV - minV) / (maxV - minV), 0.9) * 255; //0.9
			}
			setdatatoMat(out, i, j, outV, 1);
		}
	}
	return true;
}

bool CReadImage::process_img()
{
	if(in_im.empty())
		return false;
	if (in_im.depth() == CV_16U)
	{
		double minVal, maxVal;
		compute_thresh(in_im, m_coef, minVal, maxVal);
		convert_tif(in_im, out_im, minVal, maxVal);
	}
	else
	{
		if (in_im.channels() == 3)
			cvtColor(in_im, out_im, COLOR_BGR2RGB);
		else
			out_im = in_im;
	}
	return true;
}
