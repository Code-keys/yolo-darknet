#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "myutils.h"

using namespace std;
using namespace cv;

//https://blog.csdn.net/qq_33507306/article/details/104427134
//https://blog.csdn.net/gulingfengze/article/details/92013360?utm_source=app
//https://blog.csdn.net/yanfeng1022/article/details/106482923?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.not_use_machine_learn_pai&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.not_use_machine_learn_pai
//https://blog.csdn.net/zzz_zzz12138/article/details/109190019?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-3&spm=1001.2101.3001.4242

class classifier
{
public:
	classifier()
	{
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
		load_module = torch::jit::load(modelfile);
		assert(load_module != nullptr);
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
		assert(inex >= 0);
		return classname[index];
	}
private:
	torch::jit::script::Module load_module;
	torch::Tensor img_tensor;
	torch::Tensor output;
	torch::Tensor mv, sv;
	cv::Mat src;
	vector<string> classname;
};



int main(int argc, const char*argv[]) {
	if (argc < 4)
		return 0;
	classifier myclassifier;
	std::vector<string> images;
	cv::Mat src;
	myclassifier.loadmodel(argv[1]);
	myclassifier.loadclass(argv[2]);
	find_directory(argv[3], "image", images);
	for (int i = 0; i < images.size(); i++)
	{
		string filename = argv[3];
		filename = filename + "/" + images[i];
		long t1 = GetTickCount();
		src = imread(filename, cv::IMREAD_COLOR);
		myclassifier.preprocess(src, 224);
		int ret = myclassifier.predict();
		long t2 = GetTickCount();
		printf("%s (%d*%d): %s -- Elapsed Time: %dms\n", filename.c_str(), src.cols, src.rows, myclassifier.getlabel(ret).c_str(), t2 - t1);
	}
}