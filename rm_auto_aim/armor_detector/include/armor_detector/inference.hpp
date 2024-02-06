#ifndef INFERENCE_H
#define INFERENCE_H

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
    cv::Point2f top,bottom;
};

struct Detection
{
    int class_id{0};
    float confidence{0.0};
    cv::Scalar color{};
    light left_light, right_light;
    cv::Point2f center;
    int color;
    string number;
};

//创建类别到数字的映射字典
map<string,string> class_to_num;
class_to_num["0"] = "guard";
class_to_num["1"] = "1";
class_to_num["2"] = "2";
class_to_num["3"] = "3";
class_to_num["4"] = "4";
class_to_num["5"] = "5";
class_to_num["6"] = "outpost";
class_to_num["7"] = "base";
class_to_num["8"] = "1";
class_to_num["9"] = "guard";
class_to_num["10"] = "1";
class_to_num["11"] = "2";
class_to_num["12"] = "3";
class_to_num["13"] = "4";
class_to_num["14"] = "5";
class_to_num["15"] = "outpost";
class_to_num["16"] = "base";
class_to_num["17"] = "1";
class_to_num["18"] = "guard";
class_to_num["19"] = "1";
class_to_num["20"] = "2";
class_to_num["21"] = "3";
class_to_num["22"] = "4";
class_to_num["23"] = "5";
class_to_num["24"] = "outpost";
class_to_num["25"] = "base";
class_to_num["26"] = "1";
class_to_num["27"] = "guard";
class_to_num["28"] = "1";
class_to_num["29"] = "2";
class_to_num["30"] = "3";
class_to_num["31"] = "4";
class_to_num["32"] = "5";
class_to_num["33"] = "outpost";
class_to_num["34"] = "base";
class_to_num["35"] = "1";

//创建类别到颜色的映射字典
map<string,int> class_to_color;
class_to_num["0"] = BLUE;
class_to_num["1"] = BLUE;
class_to_num["2"] = BLUE;
class_to_num["3"] = BLUE;
class_to_num["4"] = BLUE;
class_to_num["5"] = BLUE;
class_to_num["6"] = BLUE;
class_to_num["7"] = BLUE;
class_to_num["8"] = BLUE;
class_to_num["9"] = RED;
class_to_num["10"] = RED;
class_to_num["11"] = RED;
class_to_num["12"] = RED;
class_to_num["13"] = RED;
class_to_num["14"] = RED;
class_to_num["15"] = RED;
class_to_num["16"] = RED;
class_to_num["17"] = RED;
class_to_num["18"] = NONE;
class_to_num["19"] = NONE;
class_to_num["20"] = NONE;
class_to_num["21"] = NONE;
class_to_num["22"] = NONE;
class_to_num["23"] = NONE;
class_to_num["24"] = NONE;
class_to_num["25"] = NONE;
class_to_num["26"] = NONE;
class_to_num["27"] = NONE;
class_to_num["28"] = NONE;
class_to_num["29"] = NONE;
class_to_num["30"] = NONE;
class_to_num["31"] = NONE;
class_to_num["32"] = NONE;
class_to_num["33"] = NONE;
class_to_num["34"] = NONE;
class_to_num["35"] = NONE;

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

    cv::Size2f modelShape{};

    float modelConfidenceThreshold {0.25};
    float modelScoreThreshold      {0.45};
    float modelNMSThreshold        {0.50};

    bool letterBoxForSquare = true;

    cv::dnn::Net net;
};
}
#endif // INFERENCE_H
