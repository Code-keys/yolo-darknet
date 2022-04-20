#pragma once
# pragma warning (disable:4819)
#include <tchar.h>
#include <windows.h>
#include <string.h>
#include <locale.h>
#include <fstream>
#include<vector>
#include <algorithm>
#include<opencv2\opencv.hpp>
#include "tinystr.h"
#include "tinyxml.h"

using namespace std;
using namespace cv;

void setdatatoMat(Mat& in, int x, int y, double v, int index);
double getdatafromMat(Mat& in, int x, int y, int index);
float compute_iou(Rect& r1, Rect& r2);
bool find_directory(const char* lpPath, const string postfix, vector<string>& image_lists);
string WcharToChar(const wchar_t* wp, size_t m_encode = CP_ACP);
wstring CharToWchar(const char* c, size_t m_encode = CP_ACP);
string ws2s(const wstring& ws);
wstring s2ws(const string& s);
string getpath();
string to_utf8(const wchar_t* buffer, int len);
string to_utf8(const wstring& str);
