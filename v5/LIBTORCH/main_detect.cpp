#include <torch/script.h> // One-stop header.
#include <torch//torch.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "myutils.h"

using namespace std;
using namespace cv;

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

std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh = 0.5, float iou_thresh = 0.5, torch::DeviceType device_type=at::kCPU)
{
	std::vector<torch::Tensor> output;
	for (size_t i = 0; i < preds.sizes()[0]; ++i)
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
		pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
		pred.select(1, 5) = std::get<1>(max_tuple);

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
			torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
			torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
			torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
			torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
			torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
			torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
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

class detector
{
public:
	detector(float conf=0.25, float nms=0.45)
	{
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
		load_module = torch::jit::load(modelfile);
		//assert(load_module != nullptr);
		if (torch::cuda::is_available())
			device_type = at::kCUDA;
		else
			device_type = at::kCPU;

		load_module.to(device_type);
		std::cout << "load model success " << std::endl;
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
		cv::Mat src;
		src = letterbox(im, width, offset_x, offset_y, gain);
		cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
		//src.convertTo(src, CV_32FC3, 1.0f / 255.0f);
		//auto img_tensor = torch::from_blob(src.data, { 1, src.rows, src.cols, src.channels() }).to(device_type);
		//img_tensor = img_tensor.permute({ 0, 3, 1, 2 }).contiguous();
		torch::Tensor img_tensor = torch::from_blob(src.data, { src.rows, src.cols, 3 }, torch::kByte).to(device_type);
		img_tensor = img_tensor.permute({ 2,0,1 });
		img_tensor = img_tensor.toType(torch::kFloat);
		img_tensor = img_tensor.div(255);
		img_tensor = img_tensor.unsqueeze(0);
		auto preds = load_module.forward({img_tensor}).toTuple()->elements()[0].toTensor();
		cout << "detect objects success" << endl;
		std::vector<torch::Tensor> dets = non_max_suppression(preds, conf_thresh, nms_thresh, device_type);
		cout << dets << endl;
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
				cv::FONT_HERSHEY_SIMPLEX, min((right - left)/200.0, 0.5), cv::Scalar(0, 0, 255), 1);
		}
	}

private:
	torch::jit::script::Module load_module;
	torch::DeviceType device_type; // 定义设备类型
	std::vector<box> detections;
	vector<string> classnames;
	float conf_thresh, nms_thresh;
};

int main(int argc, const char*argv[]) {
	if (argc < 4)
		return 0;
	detector mydetector;
	std::vector<string> images;
	mydetector.loadmodel(argv[1]);
	mydetector.loadclass(argv[2]);
	find_directory(argv[3], "image", images);
	printf("find all images %zd\n", images.size());
	cv::namedWindow("results");
	for (size_t i = 0; i < images.size(); i++)
	{
		string filename = argv[3];
		filename = filename + "/" + images[i];
		cout << filename << endl;
		cv::Mat src = cv::imread(filename, cv::IMREAD_COLOR);
		long t1 = GetTickCount();
		std::vector<box> detections = mydetector.predict(src, 640);
		long t2 = GetTickCount();
		printf("%s (%d*%d): detect %zd objects -- Elapsed Time: %dms\n", filename.c_str(), src.cols, src.rows, detections.size(), t2 - t1);
		mydetector.draw(src, detections);
		cv::imshow("results", src);
		cv::waitKey(0);
	}
}