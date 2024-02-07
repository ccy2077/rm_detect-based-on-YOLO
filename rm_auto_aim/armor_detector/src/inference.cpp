#include "armor_detector/inference.hpp"

namespace rm_auto_aim
{
Inference::Inference(const std::string &onnxModelPath, const int &color, const cv::Size &modelInputShape)
{
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    detect_color = color;
    loadOnnxNetwork();
    
    //创建类别到数字的映射字典
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
    class_to_color["0"] = BLUE;
    class_to_color["1"] = BLUE;
    class_to_color["2"] = BLUE;
    class_to_color["3"] = BLUE;
    class_to_color["4"] = BLUE;
    class_to_color["5"] = BLUE;
    class_to_color["6"] = BLUE;
    class_to_color["7"] = BLUE;
    class_to_color["8"] = BLUE;
    class_to_color["9"] = RED;
    class_to_color["10"] = RED;
    class_to_color["11"] = RED;
    class_to_color["12"] = RED;
    class_to_color["13"] = RED;
    class_to_color["14"] = RED;
    class_to_color["15"] = RED;
    class_to_color["16"] = RED;
    class_to_color["17"] = RED;
    class_to_color["18"] = NONE;
    class_to_color["19"] = NONE;
    class_to_color["20"] = NONE;
    class_to_color["21"] = NONE;
    class_to_color["22"] = NONE;
    class_to_color["23"] = NONE;
    class_to_color["24"] = NONE;
    class_to_color["25"] = NONE;
    class_to_color["26"] = NONE;
    class_to_color["27"] = NONE;
    class_to_color["28"] = NONE;
    class_to_color["29"] = NONE;
    class_to_color["30"] = NONE;
    class_to_color["31"] = NONE;
    class_to_color["32"] = NONE;
    class_to_color["33"] = NONE;
    class_to_color["34"] = NONE;
    class_to_color["35"] = NONE;
    // loadClassesFromFile(); The classes are hard-coded for this example
}

std::vector<Detection> Inference::runInference(const cv::Mat &input)
{
    cv::Mat modelInput = input;
    if (letterBoxForSquare && modelShape.width == modelShape.height)
        modelInput = formatToSquare(modelInput);

    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, modelShape, cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])

    rows = outputs[0].size[2];
    dimensions = outputs[0].size[1];

    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);

    float *data = (float *)outputs[0].data;

    float x_factor = modelInput.cols / modelShape.width;
    float y_factor = modelInput.rows / modelShape.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
       
        float *classes_scores = data+4;

        cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > modelScoreThreshold)
        {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);

            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);

        result.color = class_to_color[classes[result.class_id]];
        if (result.color != detect_color) {continue;}

        result.number = class_to_num[classes[result.class_id]];

        //获取两个灯条的顶点和底部坐标，获取装甲板中心坐标
        result.left_light.top = (cv::Point2f)boxes[idx].tl();
        result.left_light.bottom = (cv::Point2f)boxes[idx].tl() + cv::Point2f(0,boxes[idx].height);
        result.right_light.bottom = (cv::Point2f)boxes[idx].br();
        result.right_light.top = (cv::Point2f)boxes[idx].br() - cv::Point2f(0,boxes[idx].height);
        result.center = cv::Point2f(boxes[idx].x+(boxes[idx].width/2),boxes[idx].y+(boxes[idx].height/2));

        //判断装甲板大小
        float center_distance = cv::norm(((result.left_light.top+result.left_light.bottom)/2) - ((result.right_light.top+result.right_light.bottom)/2)) / boxes[idx].height;
        result.type = center_distance > 3.2 ? ArmorType::LARGE : ArmorType::SMALL;

        armors.push_back(result);
    }

    return armors;
}

void Inference::drawArmor(cv::Mat & img)
{
    for (const auto & armor : armors)
    {
        //画出装甲板
        cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
        cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
        cv::line(img, armor.left_light.top, armor.left_light.bottom, cv::Scalar(0, 255, 0), 2);
        cv::line(img, armor.right_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
        // cv::circle(img, armor.left_light.top, 3, cv::Scalar(255, 255, 255), 1);
        // cv::circle(img, armor.left_light.bottom, 3, cv::Scalar(255, 255, 255), 1);
        // cv::circle(img, armor.right_light.top, 3, cv::Scalar(255, 255, 255), 1);
        // cv::circle(img, armor.right_light.bottom, 3, cv::Scalar(255, 255, 255), 1);

        //标出置信度和类别
        cv::putText(img, std::to_string(armor.confidence), armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,cv::Scalar(0, 255, 255), 2);
        cv::putText(img, armor.number, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,cv::Scalar(0, 255, 255), 2);
    }
}

void Inference::loadOnnxNetwork()
{
    net = cv::dnn::readNetFromONNX(modelPath);

    std::cout << "\nRunning on CPU" << std::endl;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

}

cv::Mat Inference::formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}
}