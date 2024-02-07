#ifndef ARMOR_DETECTOR__INFERENCE_HPP_
#define ARMOR_DETECTOR__INFERENCE_HPP_

// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>
//STL
#include <vector>
#include <map>
// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

namespace rm_auto_aim
{
const int RED = 0;
const int BLUE = 1;
const int NONE = -1;

struct light
{
    cv::Point2f top, bottom;
    double width, height;
};

enum class ArmorType {SMALL, LARGE};

struct Detection
{
    int class_id{0};
    float confidence{0.0};
    light left_light, right_light;
    cv::Point2f center;
    int color;
    std::string number;
    ArmorType type;
};




class Inference
{
public:
    Inference(const std::string &onnxModelPath, const int &color, const cv::Size &modelInputShape = {640, 640});
    std::vector<Detection> runInference(const cv::Mat &input);
    void drawArmor(cv::Mat & img);
    std::vector<Detection> armors;
    int detect_color;

private:
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);

    std::string modelPath{};

    std::vector<std::string> classes{"0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35"};
    
    //创建类别到数字的映射字典
    std::map<std::string,std::string> class_to_num;
    //创建类别到颜色的映射字典
    std::map<std::string,int> class_to_color;
    
    cv::Size2f modelShape{};

    float modelConfidenceThreshold {0.25};
    float modelScoreThreshold      {0.45};
    float modelNMSThreshold        {0.50};

    bool letterBoxForSquare = true;

    cv::dnn::Net net;
};
}
#endif // INFERENCE_H
