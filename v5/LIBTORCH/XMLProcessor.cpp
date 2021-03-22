#include "XMLProcessor.h"
CXMLProcessor::CXMLProcessor()
{
}


CXMLProcessor::~CXMLProcessor()
{
}

//https://blog.csdn.net/weixin_40087851/article/details/83445134
bool CXMLProcessor::read_voc(const char* xmlpath, const char* filename, vector<bbox_t>& objs)
{
	vector<string> objname = objects_names_from_file(filename);
	bool ret = read_voc(xmlpath, objname, objs);
	objname.clear();
	return ret;
}

//http://blog.sina.com.cn/s/blog_69e905cd0100ks5v.html
bool CXMLProcessor::save_gaofen_new(const char* xmlPath, const char* filename, const char* classfile, vector<bbox_t>&results)
{
	char sbuf[256];
	TiXmlDocument *pDoc = new TiXmlDocument; //定义一个文档的指针											 
	TiXmlDeclaration *pDeclaration = new TiXmlDeclaration("1.0", "UTF-8", ""); //添加一个xml头。
	pDoc->LinkEndChild(pDeclaration);
	//添加XMl的根节点
	TiXmlElement *root = new TiXmlElement("Research");
	root->SetAttribute("ImageName", filename);

	wchar_t wsbuf[56];
	wstring wstr;
	string str;
	wsprintf(wsbuf,L"高分软件大赛");
	wstr = wsbuf;
	str = to_utf8(wstr);
	root->SetAttribute("Direction", str.c_str());
	pDoc->LinkEndChild(root);

	TiXmlNode * department = new TiXmlElement("Department");
	wsprintf(wsbuf, L"杭州电子科技大学");
	wstr = wsbuf;
	str = to_utf8(wstr);
	root->InsertEndChild(*department)->InsertEndChild(TiXmlText(str.c_str()));

	TiXmlNode * date = new TiXmlElement("Date");
	SYSTEMTIME st;
	GetLocalTime(&st);
	sprintf(sbuf, "%d-%02d-%02d", st.wYear, st.wMonth, st.wDay);
	root->InsertEndChild(*date)->InsertEndChild(TiXmlText(sbuf));

	TiXmlNode * pluginname = new TiXmlElement("PluginName");
	wsprintf(wsbuf, L"目标识别");
	wstr = wsbuf;
	str = to_utf8(wstr);
	root->InsertEndChild(*pluginname)->InsertEndChild(TiXmlText(str.c_str()));

	TiXmlNode * pluginclass = new TiXmlElement("PluginClass");
	wsprintf(wsbuf, L"检测");
	wstr = wsbuf;
	str = to_utf8(wstr);
	root->InsertEndChild(*pluginclass)->InsertEndChild(TiXmlText(str.c_str()));

	TiXmlElement * res = new TiXmlElement("Results");
	res->SetAttribute("Coordinate", "Pixel");

	vector<string> objname = objects_names_from_file(classfile);
	for (int i = 0; i < results.size(); i++)
	{
		TiXmlNode *obj = new TiXmlElement("Object");
		res->InsertEndChild(*obj)->InsertEndChild(TiXmlText(objname[results[i].obj_id].c_str()));
		TiXmlElement *pixel = new TiXmlElement("Pixel");
		pixel->SetAttribute("Coordinate", "X and Y");
		int x = results[i].x;
		int y = results[i].y;
		int width = results[i].w;
		int height = results[i].h;
		for (int j = 0; j < 4; j++)
		{
			TiXmlElement *pt = new TiXmlElement("Pt");
			pt->SetAttribute("RightBottomY", "");
			pt->SetAttribute("RightBottomX", "");
			if (j == 0)
			{
				sprintf(sbuf, "%d", y);
				pt->SetAttribute("LeftTopY", sbuf);
				sprintf(sbuf, "%d", x);
				pt->SetAttribute("LeftTopX", sbuf);
			}
			else if (j == 1)
			{
				sprintf(sbuf, "%d", y+height-1);
				pt->SetAttribute("LeftTopY", sbuf);
				sprintf(sbuf, "%d", x);
				pt->SetAttribute("LeftTopX", sbuf);
			}
			else if (j == 2)
			{
				sprintf(sbuf, "%d", y+height-1);
				pt->SetAttribute("LeftTopY", sbuf);
				sprintf(sbuf, "%d", x+width-1);
				pt->SetAttribute("LeftTopX", sbuf);
			}
			else if (j == 3)
			{
				sprintf(sbuf, "%d", y);
				pt->SetAttribute("LeftTopY", sbuf);
				sprintf(sbuf, "%d", x+width-1);
				pt->SetAttribute("LeftTopX", sbuf);
			}
			sprintf(sbuf, "%d", j+1);
			pt->SetAttribute("index", sbuf);
			pixel->InsertEndChild(*pt);
		}
		res->InsertEndChild(*pixel);
	}
	root->InsertEndChild(*res);
	pDoc->SaveFile(xmlPath);
	return true;
}

bool CXMLProcessor::read_voc(const char* xmlpath, vector<string> objname, vector<bbox_t>& objs)
{
	TiXmlDocument Document;
	//读取xml文件中的参数值
	if (!Document.LoadFile(xmlpath))
	{
		cout << "can not load xml file！" << endl;
		return false;
	}
	else
		cout << "load xml file success" << endl;

	TiXmlElement* RootElement = Document.RootElement();		//根目录
	TiXmlElement* NextElement = RootElement->FirstChildElement();		//根目录下的第一个节点层
	bbox_t box;
	box.prob = 1.0;

	while (NextElement != NULL)		//判断有没有读完
	{
		if (NextElement->ValueTStr() == "object")		//读到object节点
		{
			TiXmlElement* BoxElement = NextElement->FirstChildElement();
			while (BoxElement->ValueTStr() != "name")
			{
				BoxElement = BoxElement->NextSiblingElement();
			}
			string name = BoxElement->GetText();
			vector<string>::iterator iter = find(objname.begin(), objname.end(), name);
			box.obj_id = distance(objname.begin(), iter);
			while (BoxElement->ValueTStr() != "bndbox")		//读到box节点
			{
				BoxElement = BoxElement->NextSiblingElement();
			}
			//索引到xmin节点
			TiXmlElement* xminElemeng = BoxElement->FirstChildElement();
			{
				//分别读取四个数值
				int xMin = atoi(xminElemeng->GetText());
				TiXmlElement* yminElemeng = xminElemeng->NextSiblingElement();
				int yMin = atoi(yminElemeng->GetText());
				TiXmlElement* xmaxElemeng = yminElemeng->NextSiblingElement();
				int xMax = atoi(xmaxElemeng->GetText());
				TiXmlElement* ymaxElemeng = xmaxElemeng->NextSiblingElement();
				int yMax = atoi(ymaxElemeng->GetText());
				box.x = xMin;
				box.y = yMin;
				box.w = xMax - xMin + 1;
				box.h = yMax - yMin + 1;
				//加入到向量中
				objs.push_back(box);
			}
		}
		NextElement = NextElement->NextSiblingElement();
	}
	/*
	FileStorage fs(String(xmlpath), FileStorage::READ);
	if (!fs.isOpened())
	{
	cerr << "Error: cannot open .xml file";
	return false;
	}

	FileNode objects = fs["object"];
	printf("objects\n");
	FileNodeIterator it = objects.begin(), it_end = objects.end();
	printf("total objects %d\n", objects.size());
	objs.clear();
	for (; it != it_end; ++it)
	{
	bbox_t box;
	box.x = (int)(*it)["bndbox"]["xmin"];
	box.y = (int)(*it)["bndbox"]["ymin"];
	box.w = (int)(*it)["bndbox"]["xmax"] - (int)(*it)["bndbox"]["xmin"] + 1;
	box.h = (int)(*it)["bndbox"]["ymax"] - (int)(*it)["bndbox"]["ymin"] + 1;
	box.prob = 1.0;
	string name = (*it)["name"];
	vector<string>::iterator iter = find(objname.begin(), objname.end(), name);
	box.obj_id = distance(objname.begin(), iter);
	objs.push_back(box);
	}
	*/
	return true;
}

bool CXMLProcessor::save_xml(const char* xmlPath, const char* filename, vector<bbox_t>&results)
{
	wchar_t szPath[1024];
	ofstream ofs(xmlPath);
	wstring wstr;
	wsprintf(szPath, L"<?xml version=\"1.0\" encoding=\"utf-8\"?>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	wsprintf(szPath, L"<Research Direction=\"高分软件大赛\" ImageName=\"%s\">", CharToWchar(filename));
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	wsprintf(szPath, L"<Department>杭州电子科技大学</Department>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	SYSTEMTIME st;
	GetLocalTime(&st);
	wsprintf(szPath, L"<Date>%d-%02d-%02d</Date>", st.wYear, st.wMonth, st.wDay);
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	wsprintf(szPath, L"<PluginName>目标识别</PluginName>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	wsprintf(szPath, L"<PluginClass>检测</PluginClass>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	wsprintf(szPath, L"<Results Coordinate=\"Pixel\">");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	for (int i = 0; i < results.size(); i++)
	{
		wsprintf(szPath, L"<Object>Ship</Object>");
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		wsprintf(szPath, L"<Pixel Coordinate=\"X and Y\">");
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		int x = results[i].x;
		int y = results[i].y;
		int width = results[i].w;
		int height = results[i].h;
		wsprintf(szPath, L"<Pt index=\"%d\" LeftTopX=\"%d\" LeftTopY=\"%d\" RightBottomX=\"\" RightBottomY=\"\"/>", 1, x, y);
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		wsprintf(szPath, L"<Pt index=\"%d\" LeftTopX=\"%d\" LeftTopY=\"%d\" RightBottomX=\"\" RightBottomY=\"\"/>", 2, x, y + height - 1);
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		wsprintf(szPath, L"<Pt index=\"%d\" LeftTopX=\"%d\" LeftTopY=\"%d\" RightBottomX=\"\" RightBottomY=\"\"/>", 3, x + width - 1, y + height - 1);
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		wsprintf(szPath, L"<Pt index=\"%d\" LeftTopX=\"%d\" LeftTopY=\"%d\" RightBottomX=\"\" RightBottomY=\"\"/>", 4, x + width - 1, y);
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		wsprintf(szPath, L"</Pixel>");
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
	}
	wsprintf(szPath, L"</Results>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	wsprintf(szPath, L"</Research>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	ofs.close();
	return true;
}

bool CXMLProcessor::save_gaofen(const char* xmlPath, const char* filename, const char* classfile, vector<bbox_t>&results)
{
	wchar_t szPath[1024];
	ofstream ofs(xmlPath);
	wstring wstr;
	wsprintf(szPath, L"<?xml version=\"1.0\" encoding=\"utf-8\"?>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	wsprintf(szPath, L"<Research Direction=\"高分软件大赛\" ImageName=\"%s\">", CharToWchar(filename).c_str());
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	wsprintf(szPath, L"<Department>杭州电子科技大学</Department>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	SYSTEMTIME st;
	GetLocalTime(&st);
	wsprintf(szPath, L"<Date>%d-%02d-%02d</Date>", st.wYear, st.wMonth, st.wDay);
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	wsprintf(szPath, L"<PluginName>目标识别</PluginName>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	wsprintf(szPath, L"<PluginClass>检测</PluginClass>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	wsprintf(szPath, L"<Results Coordinate=\"Pixel\">");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	vector<string> objname = objects_names_from_file(classfile);
	for (int i = 0; i < results.size(); i++)
	{
		wsprintf(szPath, L"<Object>%s</Object>", CharToWchar(objname[results[i].obj_id].c_str()).c_str());
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		wsprintf(szPath, L"<Pixel Coordinate=\"X and Y\">");
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		int x = results[i].x;
		int y = results[i].y;
		int width = results[i].w;
		int height = results[i].h;
		wsprintf(szPath, L"<Pt index=\"%d\" LeftTopX=\"%d\" LeftTopY=\"%d\" RightBottomX=\"\" RightBottomY=\"\"/>", 1, x, y);
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		wsprintf(szPath, L"<Pt index=\"%d\" LeftTopX=\"%d\" LeftTopY=\"%d\" RightBottomX=\"\" RightBottomY=\"\"/>", 2, x, y + height - 1);
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		wsprintf(szPath, L"<Pt index=\"%d\" LeftTopX=\"%d\" LeftTopY=\"%d\" RightBottomX=\"\" RightBottomY=\"\"/>", 3, x + width - 1, y + height - 1);
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		wsprintf(szPath, L"<Pt index=\"%d\" LeftTopX=\"%d\" LeftTopY=\"%d\" RightBottomX=\"\" RightBottomY=\"\"/>", 4, x + width - 1, y);
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		wsprintf(szPath, L"</Pixel>");
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
	}
	wsprintf(szPath, L"</Results>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	wsprintf(szPath, L"</Research>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	ofs.close();
	return true;
}

vector<string> CXMLProcessor::objects_names_from_file(const string filename)
{
	ifstream file(filename);
	vector<string> file_lines;
	if (!file.is_open()) return file_lines;
	for (string line; getline(file, line);) file_lines.push_back(line);
	cout << "object names loaded \n";
	return file_lines;
}

bool CXMLProcessor::save_voc(const char* xmlPath, const char* filename, const char* classname, Mat& img, vector<bbox_t>&results)
{
	wchar_t szPath[1024];
	ofstream ofs(xmlPath);
	wstring wstr;
	wsprintf(szPath, L"<?xml version=\"1.0\"?>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;

	wsprintf(szPath, L"<annotation verified=\"no\">");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;

	wsprintf(szPath, L"<path>%s</path>", CharToWchar(filename).c_str());
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;

	wsprintf(szPath, L"<source>\n<database>Unknown</database>\n</source>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;

	wsprintf(szPath, L"<size>\n<width>%d</width>\n<height>%d</height>\n<depth>%d</depth>\n</size>", img.cols, img.rows, img.channels());
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;

	wsprintf(szPath, L"<segmented>0</segmented>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;

	vector<string> objname = objects_names_from_file(classname);

	for (int i = 0; i < results.size(); i++)
	{
		wsprintf(szPath, L"<object>");
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
		wsprintf(szPath, L"<name>%s</name>", CharToWchar(objname[results[i].obj_id].c_str()).c_str());
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;

		wsprintf(szPath, L"<pose>Unspecified</pose>\n<truncated>0</truncated>\n<difficult>0</difficult>\n<bndbox>");
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;

		int x = results[i].x;
		int y = results[i].y;
		int width = results[i].w;
		int height = results[i].h;
		wsprintf(szPath, L"<xmin>%d</xmin>\n<ymin>%d</ymin>\n<xmax>%d</xmax>\n<ymax>%d</ymax>", x, y, x + width - 1, y + height - 1);
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;

		wsprintf(szPath, L"</bndbox>\n</object>");
		wstr = szPath;
		ofs << to_utf8(wstr) << endl;
	}
	wsprintf(szPath, L"</annotation>");
	wstr = szPath;
	ofs << to_utf8(wstr) << endl;
	ofs.close();
	return true;
}

bool CXMLProcessor::save_txt(const char* xmlPath, const char* filename, const char* classname, vector<bbox_t>&results)
{
	ofstream ofs(xmlPath);
	vector<string> objname = objects_names_from_file(classname);
	char sbuf[256];
	for (int i = 0; i < results.size(); i ++)
	{
		bbox_t& r = results[i];
		sprintf(sbuf, "%d,%d,%d,%d,%s\n", r.x, r.y, r.x + r.w - 1, r.y + r.h - 1, objname[results[i].obj_id].c_str());
		ofs << sbuf;
	}
	ofs.close();
	return true;
}