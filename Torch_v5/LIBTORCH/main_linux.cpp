#include <torch/script.h> // One-stop header.
#include <torch//torch.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
//#include "myutils.h"

#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <stdio.h>

using namespace std;
using namespace cv;

//https://www.cnblogs.com/yanghailin/p/12901586.html
//https://blog.csdn.net/weixin_42398658/article/details/111954760
//https://blog.csdn.net/weixin_42398658/article/details/112602722
//https://blog.csdn.net/weixin_44936889/article/details/111186818

typedef struct box
{
	int id;
	int left;
	int right;
	int top;
	int bottom;
	float score;
	int classID;
	int reserved;
}box;


class detector
{
public:
	detector(float conf=0.15, float nms=0.5)
	{
		//LoadLibraryA("ATen_cuda.dll");
		//LoadLibraryA("c10_cuda.dll");
		//LoadLibraryA("torch_cuda.dll");
		//LoadLibraryA("torchvision.dll");
		device_type = at::kCUDA;
		conf_thresh = conf;
		nms_thresh = nms;
	}
	~detector()
	{
		classnames.clear();
		detections.clear();
	}
public:
	void loadmodel(const char* modelfile)
	{
		try
		{
			load_module = torch::jit::load(modelfile);
			//assert(load_module != nullptr);
		}
		catch (const c10::Error& e) {
			std::cout << e.msg();
			std::cerr << "error loading the model \n";
		}

		if (torch::cuda::is_available())
		{
			device_type = at::kCUDA;
			std::cout << "cuda" << std::endl;
		}
		else
		{
			device_type = at::kCPU;
			std::cout << "cpu" << std::endl;
		}

		load_module.to(device_type);
		std::cout << "load model success " << std::endl;

		cv::Mat re(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
		predict(re, 640);
		predict(re, 640);
	}
	void loadclass(const char* classfile)
	{
		ifstream file(classfile);
		if (!file.is_open()) return;
		for (string line; getline(file, line);) classnames.push_back(line);
		cout << "object names loaded " << classnames.size() << endl;
	}
	cv::Mat letterbox(cv::Mat& src, int width, int& x, int& y, float& gain)
	{
		int w, h;
		float r_w = width / (src.cols * 1.0);
		float r_h = width / (src.rows * 1.0);
		if (r_h > r_w) {
			w = width;
			h = r_w * src.rows;
			x = 0;
			y = (width - h) / 2;
			gain = r_w;
		}
		else {
			w = r_h * src.cols;
			h = width;
			x = (width - w) / 2;
			y = 0;
			gain = r_h;
		}
		cv::Mat re(h, w, CV_8UC3);
		cv::resize(src, re, re.size(), 0, 0, cv::INTER_LINEAR);
		cv::Mat out(width, width, CV_8UC3, cv::Scalar(128, 128, 128));
		re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
		return out;
	}
	std::vector<box> predict(cv::Mat& im, int width=640)
	{
		int offset_x, offset_y;
		float gain = 0;
		cv::Mat src = letterbox(im, width, offset_x, offset_y, gain);
		cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
		//src.convertTo(src, CV_32FC3, 1.0f / 255.0f);
		//torch::Tensor img_tensor = torch::from_blob(src.data, { 1, src.rows, src.cols, src.channels() }).to(device_type);
		//img_tensor = img_tensor.permute({ 0, 3, 1, 2 }).contiguous();
		img_tensor = torch::from_blob(src.data, { src.rows, src.cols, 3 }, torch::kByte).to(device_type);
		img_tensor = img_tensor.permute({ 2,0,1 });
		img_tensor = img_tensor.toType(torch::kFloat);
		img_tensor = img_tensor.div(255);
		img_tensor = img_tensor.unsqueeze(0);
		preds = load_module.forward({img_tensor}).toTuple()->elements()[0].toTensor();
		//cout << "detect objects success" << endl;
		//std::vector<torch::Tensor> dets = non_max_suppression(preds, conf_thresh, nms_thresh, device_type);
		std::vector<torch::Tensor> dets = new_nms(preds, conf_thresh, nms_thresh, device_type);
		detections.clear();
		if (dets.size() > 0)
		{
			box _det;
			for (size_t i = 0; i < dets[0].sizes()[0]; ++i)
			{
				float left = (dets[0][i][0].item().toFloat() - offset_x) / gain;
				float top = (dets[0][i][1].item().toFloat() - offset_y) / gain;
				float right = (dets[0][i][2].item().toFloat() - offset_x) / gain;
				float bottom = (dets[0][i][3].item().toFloat() - offset_y) / gain;
				float score = dets[0][i][4].item().toFloat();
				int classID = dets[0][i][5].item().toInt();
				_det.left = (int)left;
				_det.top = (int)top;
				_det.right = (int)right;
				_det.bottom = (int)bottom;
				_det.id = (int)i;
				_det.score = score;
				_det.classID = classID;
				_det.reserved = -1;
				detections.push_back(_det);
			}
		}
		dets.clear();
		return detections;
	}
	void draw(cv::Mat& im, std::vector<box>& detections)
	{
		for (size_t i = 0; i < detections.size(); i++)
		{
			int left = detections[i].left;
			int right = detections[i].right;
			int top = detections[i].top;
			int bottom = detections[i].bottom;
			int classID = detections[i].classID;
			float score = detections[i].score;
			cv::rectangle(im, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 0, 255), 2);

			cv::putText(im, classnames[classID] + ": " + cv::format("%.2f", score), cv::Point(left, top),
				cv::FONT_HERSHEY_SIMPLEX, min((right-left)/200.0, 0.5), cv::Scalar(0, 0, 255), 1);
		}
	}
private:
	bool nms(const torch::Tensor& boxes, const torch::Tensor& scores, torch::Tensor& keep, int& count, float overlap, int top_k)
	{
		count = 0;
		keep = torch::zeros({ scores.size(0) }).to(torch::kLong).to(scores.device());
		if (0 == boxes.numel())
		{
			return false;
		}

		torch::Tensor x1 = boxes.select(1, 0).clone();
		torch::Tensor y1 = boxes.select(1, 1).clone();
		torch::Tensor x2 = boxes.select(1, 2).clone();
		torch::Tensor y2 = boxes.select(1, 3).clone();
		torch::Tensor area = (x2 - x1) * (y2 - y1);
		//    std::cout<<area<<std::endl;

		std::tuple<torch::Tensor, torch::Tensor> sort_ret = torch::sort(scores.unsqueeze(1), 0, 0);
		torch::Tensor v = std::get<0>(sort_ret).squeeze(1).to(scores.device());
		torch::Tensor idx = std::get<1>(sort_ret).squeeze(1).to(scores.device());

		int num_ = idx.size(0);
		if (num_ > top_k) //python:idx = idx[-top_k:]
		{
			idx = idx.slice(0, num_ - top_k, num_).clone();
		}
		torch::Tensor xx1, yy1, xx2, yy2, w, h;
		while (idx.numel() > 0)
		{
			auto i = idx[-1];
			keep[count] = i;
			count += 1;
			if (1 == idx.size(0))
			{
				break;
			}
			idx = idx.slice(0, 0, idx.size(0) - 1).clone();

			xx1 = x1.index_select(0, idx);
			yy1 = y1.index_select(0, idx);
			xx2 = x2.index_select(0, idx);
			yy2 = y2.index_select(0, idx);

			xx1 = xx1.clamp(x1[i].item().toFloat(), INT_MAX * 1.0);
			yy1 = yy1.clamp(y1[i].item().toFloat(), INT_MAX * 1.0);
			xx2 = xx2.clamp(INT_MIN * 1.0, x2[i].item().toFloat());
			yy2 = yy2.clamp(INT_MIN * 1.0, y2[i].item().toFloat());

			w = xx2 - xx1;
			h = yy2 - yy1;

			w = w.clamp(0, INT_MAX);
			h = h.clamp(0, INT_MAX);

			torch::Tensor inter = w * h;
			torch::Tensor rem_areas = area.index_select(0, idx);

			torch::Tensor union_ = (rem_areas - inter) + area[i];
			torch::Tensor Iou = inter * 1.0 / union_;
			torch::Tensor index_small = Iou < overlap;
			auto mask_idx = torch::nonzero(index_small).squeeze();
			idx = idx.index_select(0, mask_idx);//pthon: idx = idx[IoU.le(overlap)]
		}
		return true;
	}

	std::vector<torch::Tensor> new_nms(const torch::Tensor& preds, float score_thresh = 0.5, float iou_thresh = 0.5, torch::DeviceType device_type = at::kCPU)
	{
		//cenx, ceny , w, h, score, classes, 
		std::vector<torch::Tensor> output;
		for (size_t i = 0; i < preds.sizes()[0]; ++i) // 3 layers
		{
			torch::Tensor pred = preds.select(0, i);
			// Filter by scores
			torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));

			pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
			if (pred.sizes()[0] == 0) continue;

			// (center_x, center_y, w, h) to (left, top, right, bottom)
			pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
			pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
			pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
			pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

			// Computing scores and classes
			std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
			pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple); //score
			pred.select(1, 5) = std::get<1>(max_tuple);  // classes

			torch::Tensor  dets = pred.slice(1, 0, 6);
			torch::Tensor boxes_ = pred.slice(1, 0, 4);
			torch::Tensor scores_ = pred.select(1, 4);
			torch::Tensor keep;
			int count;
			nms(boxes_, scores_, keep, count, 0.5, dets.sizes()[0]);

			keep = keep.toType(torch::kInt64);
			output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)).to(at::kCPU));
		}
		return output;
	}

	std::vector<torch::Tensor> non_max_suppression(const torch::Tensor& preds, float score_thresh = 0.5, float iou_thresh = 0.5, torch::DeviceType device_type = at::kCPU)
	{
		//cenx, ceny , w, h, score, classes, 
		std::vector<torch::Tensor> output;
		for (size_t i = 0; i < preds.sizes()[0]; ++i) // 3 layers
		{
			torch::Tensor pred = preds.select(0, i);
			// Filter by scores
			torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));

			pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
			if (pred.sizes()[0] == 0) continue;

			// (center_x, center_y, w, h) to (left, top, right, bottom)
			pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
			pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
			pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
			pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

			// Computing scores and classes
			std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
			pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple); //score
			pred.select(1, 5) = std::get<1>(max_tuple);  // classes

			torch::Tensor  dets = pred.slice(1, 0, 6);

			torch::Tensor keep = torch::empty({ dets.sizes()[0] }).to(device_type);
			torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
			std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
			torch::Tensor v = std::get<0>(indexes_tuple);
			torch::Tensor indexes = std::get<1>(indexes_tuple);
			int count = 0;
			while (indexes.sizes()[0] > 0)
			{
				keep[count] = (indexes[0].item().toInt());
				count += 1;

				// Computing overlaps
				torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1).to(device_type);
				torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1).to(device_type);
				torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1).to(device_type);
				torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1).to(device_type);
				torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1).to(device_type);
				torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1).to(device_type);
				for (size_t i = 0; i < indexes.sizes()[0] - 1; ++i)
				{
					lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
					tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
					rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
					bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
					widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
					heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
				}
				torch::Tensor overlaps = (widths * heights).to(device_type);

				// FIlter by IOUs
				torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
				indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
			}
			keep = keep.toType(torch::kInt64);
			output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)).to(at::kCPU));
		}
		return output;
	}
private:
	torch::jit::script::Module load_module;
	torch::DeviceType device_type; 
	torch::Tensor img_tensor;
	torch::Tensor preds;
	std::vector<box> detections;
	vector<string> classnames;
	float conf_thresh, nms_thresh;
};

//https://blog.csdn.net/qq_33507306/article/details/104427134
//https://blog.csdn.net/gulingfengze/article/details/92013360?utm_source=app
//https://blog.csdn.net/yanfeng1022/article/details/106482923?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.not_use_machine_learn_pai&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.not_use_machine_learn_pai
//https://blog.csdn.net/zzz_zzz12138/article/details/109190019?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-3&spm=1001.2101.3001.4242

class classifier
{
public:
	classifier()
	{
		//LoadLibraryA("c10_cuda.dll");
		//LoadLibraryA("torch_cuda.dll");
		mv = torch::zeros({ 1,3 });
		mv[0][0] = 0.485;
		mv[0][1] = 0.456;
		mv[0][2] = 0.406;
		mv = mv.cuda();
		//std::cout << mv << std::endl;
		sv = torch::zeros({ 1,3 });
		sv[0][0] = 0.229;
		sv[0][1] = 0.224;
		sv[0][2] = 0.225;
		sv = sv.cuda();
		//std::cout << sv << std::endl;
	}
	~classifier()
	{
		if (!src.empty())
			src.release();
		classname.clear();
	}
public:
	void loadmodel(const char* modelfile)
	{
		std::cout << "load model" << std::endl;
		try
		{
			load_module = torch::jit::load(modelfile);
		}
		catch (const c10::Error& e) {
			std::cout << e.msg();
			std::cerr << "error loading the model \n";
		}

		if (torch::cuda::is_available())
		{
			device_type = at::kCUDA;
			std::cout << "cuda" << std::endl;
		}
		else
		{
			device_type = at::kCPU;
			std::cout << "cpu" << std::endl;
		}

		load_module.to(device_type);
//		assert(load_module != nullptr);
		std::cout << "load model success " << std::endl;
	}
	void loadclass(const char* classfile)
	{
		ifstream file(classfile);
		if (!file.is_open()) return;
		for (string line; getline(file, line);) classname.push_back(line);
		cout << "object names loaded " << classname.size() << endl;
	}
	void preprocess(cv::Mat& im, int width)
	{
		cv::cvtColor(im, src, cv::COLOR_BGR2RGB);
		cv::resize(src, src, cv::Size(width, width));
		img_tensor = torch::from_blob(src.data, { src.rows, src.cols, 3 }, torch::kByte);
		img_tensor = img_tensor.cuda();

		img_tensor = img_tensor.toType(torch::kFloat).div(255);
		img_tensor = img_tensor.subtract(mv);
		img_tensor = img_tensor.div(sv);

		img_tensor = img_tensor.permute({ 2, 0, 1 });
		img_tensor = img_tensor.unsqueeze(0);
	}
	int predict()
	{
		output = load_module.forward({ img_tensor }).toTensor();
		auto max_result = output.max(1, true);
		int max_index = std::get<1>(max_result).item<float>();
		//	std::cout << "result: " << max_index << std::endl;
		return max_index;
	}
	string getlabel(int index)
	{
		//assert(inex >= 0);
		return classname[index];
	}
private:
	torch::jit::script::Module load_module;
	torch::DeviceType device_type;
	torch::Tensor img_tensor;
	torch::Tensor output;
	torch::Tensor mv, sv;
	cv::Mat src;
	vector<string> classname;
};

enum DETECTORRECOG
{
	DETECTOR,
	RECOGNITION,
	TEST
};



std::string& trim(std::string &s)
{
	if (s.empty())
	{
		return s;
	}
 
	s.erase(0,s.find_first_not_of(" "));
	s.erase(s.find_last_not_of(" ") + 1);

	return s;
}

//https://blog.csdn.net/chenyijun/article/details/52484803?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242

//struct dirent
//{
//   long d_ino;                 /* inode number 索引节点号 */
//    off_t d_off;                /* offset to this dirent 在目录文件中的偏移 */
//    unsigned short d_reclen;    /* length of this d_name 文件名长 */
//   unsigned char d_type;        /* the type of d_name 文件类型 */    
//    char d_name [NAME_MAX+1];   /* file name (null-terminated) 文件名，最长255字符 */
//};
 
//struct __dirstream
//{
//    void *__fd;                        /* `struct hurd_fd' pointer for descriptor.  */
//    char *__data;                /* Directory block.  */
//    int __entry_data;                /* Entry number `__data' corresponds to.  */
//    char *__ptr;                /* Current pointer into the block.  */
//    int __entry_ptr;                /* Entry number `__ptr' corresponds to.  */
//    size_t __allocation;        /* Space allocated for the block.  */
//    size_t __size;                /* Total valid data in the block.  */
//    __libc_lock_define (, __lock) /* Mutex lock for this structure.  */
//};

//typedef struct __dirstream DIR; 


void find_directory(const string foldername, vector<string>& images)
{
	DIR  *dir;
    struct  dirent  *ptr;
    dir = opendir(foldername.c_str()); ///open the dir
    string str;
	images.clear();
    while((ptr = readdir(dir)) != NULL) ///read the list of this dir
    {
        #ifdef _WIN32
            printf("d_name: %s\n", ptr->d_name);
            images.push_back(ptr->d_name);
        #endif
        #ifdef __linux
            printf("d_type:%d d_name: %s\n", ptr->d_type,ptr->d_name);
            if(ptr->d_type == 8) //files
				images.push_back(ptr->d_name);
        #endif
    }
    closedir(dir);
}

int main(int argc, const char*argv[]) {
	if (argc != 5)
	{
		cout << "usage: detect type *.pt *.names folder" << endl;
		return 0;
	}
	else
	{
		for (int i = 1; i < argc; i++)
			std::cout << argv[i] << " ";
		std::cout << std::endl;
	}
	int type = atoi(argv[1]);
	if (type == DETECTOR)
	{
		detector mydetector;
		std::vector<string> images;
		mydetector.loadmodel(argv[2]);
		mydetector.loadclass(argv[3]);
		find_directory(argv[4], images);
		printf("find all images %zd\n", images.size());
		cv::namedWindow("results");
		for (size_t i = 0; i < images.size(); i++)
		{
			string filename = argv[4];
			filename = filename + "/" + images[i];
			cout << filename << endl;
			cv::Mat src = cv::imread(filename, cv::IMREAD_COLOR);
			cudaDeviceSynchronize();
			auto start = std::chrono::high_resolution_clock::now();
			std::vector<box> detections = mydetector.predict(src, 640);
			auto end = std::chrono::high_resolution_clock::now();
			cudaDeviceSynchronize();
			std::chrono::duration<double> duration = end - start;
			printf("%s (%d*%d): detect %zd objects -- Elapsed Time: %fms\n", filename.c_str(), src.cols, src.rows, detections.size(), duration.count()*1000);
			mydetector.draw(src, detections);
			cv::imshow("results", src);
			cv::waitKey(0);
		}
	}
	else if (type == RECOGNITION)
	{
		classifier myclassifier;
		std::vector<string> images;
		cv::Mat src;
		myclassifier.loadmodel(argv[2]);
		myclassifier.loadclass(argv[3]);
		find_directory(argv[4], images);
		for (int i = 0; i < images.size(); i++)
		{
			string filename = argv[4];
			filename = filename + "/" + images[i];
			auto start = std::chrono::high_resolution_clock::now();
			src = imread(filename, cv::IMREAD_COLOR);
			myclassifier.preprocess(src, 224);
			int ret = myclassifier.predict();
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration = end - start;
			printf("%s (%d*%d): %s -- Elapsed Time: %fms\n", filename.c_str(), src.cols, src.rows, myclassifier.getlabel(ret).c_str(), duration.count()*1000);
		}
	}	
	else if (type == TEST)
	{
		detector mydetector;
		std::vector<string> images;
		mydetector.loadmodel(argv[2]);
		mydetector.loadclass(argv[3]);
		std::string filename = argv[4];
		std::cout << filename << " " << filename.size() << std::endl;
		cv::VideoCapture video;		
		video.open(trim(filename));
		if(!video.isOpened())
		{
		        std::cout<<"video not open."<<std::endl;
		        return 1;
		}

		cv::namedWindow("results");
	        
		while (true)
		{
			cv::Mat src;
			bool ret = video.read(src);
			if (!ret) break;
			cudaDeviceSynchronize();
			auto start = std::chrono::steady_clock::now();
			std::vector<box> detections = mydetector.predict(src, 640);
			cudaDeviceSynchronize();
			auto end = std::chrono::steady_clock::now();
			std::chrono::duration<double> duration = end - start;
			printf("%s (%d*%d): detect %zd objects -- Elapsed Time: %fms\n", argv[4], src.cols, src.rows, detections.size(), duration.count() * 1000);
			mydetector.draw(src, detections);
			cv::imshow("results", src);
			cv::waitKey(40);
		}
	}
}
