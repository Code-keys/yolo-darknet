#include "myutils.h"

double getdatafromMat(Mat& in, int x, int y, int index)
{
	if (index == 0)
	{
		float data = in.ptr<float>(x)[y];
		return double(data);
	}
	else if (index == 1)
	{
		uint16_t data = in.ptr<uint16_t>(x)[y];
		return double(data);
	}
	else if (index == 2)
	{
		uchar data = in.ptr<uchar>(x)[y];
		return double(data);
	}
	return 0;
}

void setdatatoMat(Mat& in, int x, int y, double v, int index)
{
	if (index == 0)
		in.ptr<float>(x)[y] = (float)v;
	if (index == 1)
		in.ptr<uchar>(x)[y] = (uchar)v;
}

float compute_iou(Rect& r1, Rect& r2)
{
	int x0 = max(r1.x, r2.x);
	int y0 = max(r1.y, r2.y);
	int x1 = min(r1.x + r1.width - 1, r2.x + r2.width - 1);
	int y1 = min(r1.y + r1.height - 1, r2.y + r2.height - 1);
	int w = max(x1 - x0, 0);
	int h = max(y1 - y0, 0);
	int area = w * h;
	int area1 = r1.area();
	int area2 = r2.area();
	float iou = float(area) / float(area1 + area2 - area + 1e-5);
	return iou;
}

bool find_directory(const char * lpPath, const string postfix, vector<string>& image_list)
{
	//printf("******************查找文件开始*************************\n");
	char szFile[100];
	char szFind[MAX_PATH];
	//WIN32_FIND_DATA结构描述了一个由FindFirstFile,   
	//FindFirstFileEx, 或FindNextFile函数查找到的文件信息  
	WIN32_FIND_DATA FindFileData;
	memset(szFile, 0, 100);         //为新申请的内存做初始化工作  
	strcpy(szFind, lpPath);        //将lpPath的值拷贝给szFind  
	strcat(szFind, "//*.*");       //联接构成完整路径名，双斜杠//用于组成下一级路径  
								   //通过FindFirstFile()函数根据当前的文件存放路径  
								   //查找该文件来把待操作文件的相关属性读取到WIN32_FIND_DATA结构中去 
	wstring wszPath = CharToWchar(szFind);
	HANDLE hFind = ::FindFirstFile(wszPath.c_str(), &FindFileData);
	if (INVALID_HANDLE_VALUE == hFind)
		return false;  //如果hFind的句柄值无效，返回0  

	while (TRUE)
	{
		//将dwFileAttributes和FILE_ATTRIBUTE_DIRECTORY做位"与"运算来判断所找到的项目是不是文件夹，  
		//这段程序的目的是查找文件夹下子文件夹中的内容  
		if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			if (FindFileData.cFileName[0] != '.')
			{
				strcpy(szFile, lpPath);
				strcat(szFile, "//");
				string str = WcharToChar(FindFileData.cFileName);
				strcat(szFile, str.c_str());
				find_directory(szFile, postfix, image_list);
			}
		}
		else
		{
			string filename = WcharToChar(FindFileData.cFileName);
			//printf("%s\n", filename.substr(filename.length() - 4, 4).c_str());
			//string::size_type position;
			if (postfix.compare("image") == 0)
			{
				if ((filename.find("tiff") != filename.npos && filename.find("tiff") == filename.size() - 4)
					|| (filename.find("tif") != filename.npos && filename.find("tif") == filename.size() - 3)
					|| (filename.find("jpg") != filename.npos && filename.find("jpg") == filename.size() - 3)
					|| (filename.find("bmp") != filename.npos && filename.find("bmp") == filename.size() - 3)
					|| (filename.find("jpeg") != filename.npos && filename.find("jpeg") == filename.size() - 4))
					image_list.push_back(filename);
			}
			else if (postfix.compare("xml") == 0)
			{
				if ((filename.find("xml") != filename.npos && filename.find("xml") == filename.size() - 3))
				{

					image_list.push_back(filename);
				}
			}
		}
		if (!FindNextFile(hFind, &FindFileData))
			break;//如果没有找到下一个文件，结束本次循环  
	}
	FindClose(hFind);
	//printf("******************查找文件结束*************************\n");
	return true;
}

string to_utf8(const wchar_t* buffer, int len)
{
	int nChars = ::WideCharToMultiByte(CP_UTF8, 0, buffer, len, NULL, 0, NULL, NULL);
	if (nChars == 0)return "";

	string newbuffer;
	newbuffer.resize(nChars);
	::WideCharToMultiByte( CP_UTF8, 0, buffer, len, const_cast<char*>(newbuffer.c_str()), nChars, NULL, NULL);

	return newbuffer;
}

string to_utf8(const wstring& str)
{
	return to_utf8(str.c_str(), (int)str.size());
}

wstring CharToWchar(const char* c, size_t m_encode)
{
	wstring str;
	int len = MultiByteToWideChar(m_encode, 0, c, strlen(c), NULL, 0);
	wchar_t*	m_wchar = new wchar_t[len + 1];
	MultiByteToWideChar(m_encode, 0, c, strlen(c), m_wchar, len);
	m_wchar[len] = '\0';
	str = m_wchar;
	delete m_wchar;
	return str;
}

string WcharToChar(const wchar_t* wp, size_t m_encode)
{
	string str;
	int len = WideCharToMultiByte(m_encode, 0, wp, wcslen(wp), NULL, 0, NULL, NULL);
	char	*m_char = new char[len + 1];
	WideCharToMultiByte(m_encode, 0, wp, wcslen(wp), m_char, len, NULL, NULL);
	m_char[len] = '\0';
	str = m_char;
	delete m_char;
	return str;
}

string ws2s(const wstring& ws)
{
	size_t convertedChars = 0;
	string curLocale = setlocale(LC_ALL, NULL); //curLocale="C"
	setlocale(LC_ALL, "chs");
	const wchar_t* wcs = ws.c_str();
	size_t dByteNum = sizeof(wchar_t)*ws.size() + 1;
	//cout << "ws.size():" << ws.size() << endl;            //8  “123ABC你好”共8个字符

	char* dest = new char[dByteNum];
	wcstombs_s(&convertedChars, dest, dByteNum, wcs, _TRUNCATE);
	//cout << "ws2s_convertedChars:" << convertedChars << endl; //11 共使用了11个字节存储多字节字符串 包括结束符
	string result = dest;
	delete[] dest;
	setlocale(LC_ALL, curLocale.c_str());
	return result;
}

wstring s2ws(const string& s)
{
	size_t convertedChars = 0;
	string curLocale = setlocale(LC_ALL, NULL);   //curLocale="C"
	setlocale(LC_ALL, "chs");
	const char* source = s.c_str();
	size_t charNum = sizeof(char)*s.size() + 1;
	//cout << "s.size():" << s.size() << endl;   //10 “123ABC你好”共10个字节

	wchar_t* dest = new wchar_t[charNum];
	mbstowcs_s(&convertedChars, dest, charNum, source, _TRUNCATE);
	//cout << "s2ws_convertedChars:" << convertedChars << endl; //9 转换为9个字符 包括结束符
	wstring result = dest;
	//cout << result.c_str();
	delete[] dest;
	setlocale(LC_ALL, curLocale.c_str());
	return result;
}

string getpath()
{
	wchar_t AppPath[MAX_PATH] = {0};
	char g_strAppPath[MAX_PATH] = {0};
	string path;

	::GetModuleFileName(NULL, AppPath, MAX_PATH);
	WideCharToMultiByte(CP_ACP, 0, (LPCWSTR)AppPath, -1, g_strAppPath, 256, 0, 0);
	*strrchr(g_strAppPath, '\\') = '\0';
	path = g_strAppPath;
	return path;
}