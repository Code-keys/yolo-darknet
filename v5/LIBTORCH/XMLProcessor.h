#pragma once
#include "myutils.h"

struct bbox_t {
    unsigned int x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;           // class of object - from range [0, classes-1]
    unsigned int track_id;         // tracking id for video (0 - untracked, 1 - inf - tracked object)
    unsigned int frames_counter;   // counter of frames on which the object was detected
    float x_3d, y_3d, z_3d;        // center of object (in Meters) if ZED 3D Camera is used
};

class CXMLProcessor
{
public:
	CXMLProcessor();
	~CXMLProcessor();
public:
	bool read_voc(const char* xmlpath, const char* filename, vector<bbox_t>& objs);
	bool read_voc(const char* xmlpath, vector<string> objname, vector<bbox_t>& objs);
	bool save_xml(const char* xmlPath, const char* filename, vector<bbox_t>&results);
	bool save_gaofen(const char* xmlPath, const char* filename, const char* classname, vector<bbox_t>&results);
	bool save_voc(const char* xmlPath, const char* filename, const char* classname, Mat& img, vector<bbox_t>&results);
	bool save_gaofen_new(const char* xmlPath, const char* filename, const char* classname, vector<bbox_t>&results);
	bool save_txt(const char* xmlPath, const char* filename, const char* classname, vector<bbox_t>&results);
private:
	vector<string> objects_names_from_file(const string  filename);
};

