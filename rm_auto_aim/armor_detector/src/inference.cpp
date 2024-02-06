#include "armor_detector/inference.hpp"
namespace rm_auto_aim
{
Inference::Inference(const std::string &onnxModelPath, const int &color, const cv::Size &modelInputShape = {640, 640})
{
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    detect_color = color;
    loadOnnxNetwork();
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
        result.left_light.top = boxes[idx].tl();
        result.left_light.bottom = boxes[idx].tl() + cv::point2f(0,boxes[idx].height);
        result.right_light.bottom = boxes[idx].br();
        result.right_light.top = boxes[idx].br() - cv::pointf(0,boxes[idx].height);
        result.center = cv::point2f(boxes[idx].x+(boxes[idx].weight/2),boxes[idx].y+(boxes[idx].height/2));

        armors.push_back(result);
    }

    return armors;
}
void Inferecen::drawArmor(cv::Mat & img)
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
        cv::putText(img, armor.confidence, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,cv::Scalar(0, 255, 255), 2);
        cv::putText(img, armor.className, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,cv::Scalar(0, 255, 255), 2);
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