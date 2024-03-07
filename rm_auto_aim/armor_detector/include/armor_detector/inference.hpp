#ifndef ARMOR_DETECTOR__INFERENCE_HPP_
#define ARMOR_DETECTOR__INFERENCE_HPP_

#include <fstream>
#include <sstream>
#include <iostream>

#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;


namespace rm_auto_aim
{
const int RED = 0;
const int BLUE = 1;
const int NONE = -1;

struct light
{
    Point2f top, bottom;
    double width, height;
};

enum class ArmorType {SMALL, LARGE};

struct Detection
{
    int class_id{0};
    float confidence{0.0};
    light left_light, right_light;
    Point2f center;
    int color;
    string number;
    ArmorType type;
};


class Inference
{
public:
	Inference(string modelpath, int color);
	vector<Detection> runInference(Mat& frame);
    vector<Detection> Armors;
    int detect_color;
    void drawArmor(Mat& img);
private:
	cv::Mat resize_image(Mat srcimg, int *newh, int *neww, int *padh, int *padw);
	const bool keep_ratio = true;
	const int inpWidth = 640;
	const int inpHeight = 640;
	float confThreshold;
	float nmsThreshold;
	const int num_class = 36;  //总共36个类别
	const int reg_max = 16;
    
   
     //创建类别到数字的映射字典
    std::map<int,string> class_to_num;
    //创建类别到颜色的映射字典
    std::map<int,int> class_to_color;

	cv::dnn::Net net;

	void softmax_(const float* x, float* y, int length);
	void generate_proposal(Mat out, vector<int>& cls_ids, vector<Rect>& boxes, vector<float>& confidences, vector< vector<Point>>& landmarks, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw);
	
};
}
#endif