#include "src/darknet.h"
#include <iostream>
#include "detector_class.h"
int check_mistakes = 0;

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec)
{

    for (auto &i : result_vec) {
        cv::Scalar color(255, 255, 255);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
    }
}

//int main(void) {
//    //  printf("hello world\n");
//    std::string data_file = "../data/plate.data";
//    std::string cfg_file = "../data/yolov3-tiny.cfg";
//    std::string weights_file = "../data/yolov3-tiny_last.weights";
//    std::string file_name = "/media/dat/05C830EB6380925C/data/plate_from_vehicle/"
//                            "data/half1/000777.jpg";
//#if(0)
//    char *weights = (char *)weights_file.data();
//    if (weights)
//        if (strlen(weights) > 0)
//            if (weights[strlen(weights) - 1] == 0x0d)
//                weights[strlen(weights) - 1] = 0;

//    float thresh = 0.25f;
//    float hier_thresh = 0.5f;
//    test_detector((char *)data_file.data(), (char *)cfg_file.data(), weights,
//                  (char *)file_name.data(), thresh, hier_thresh, 0, 0, 0,
//                  (char *)0, 0);
//#else
//    Detector *detector = new Detector(cfg_file, weights_file);
////    std::shared_ptr<image_t> det_image;
//    std::vector<bbox_t> result_vec;
//    const float thresh = 0.3f;
//    cv::Mat current_frame = cv::imread(file_name);
////    det_image = detector->mat_to_image_resize(current_frame);
////    result_vec = detector->detect_resized(*det_image, current_frame.size().width,
////                                     current_frame.size().height, thresh, true);
//    result_vec = detector->detect(current_frame, thresh, false);
//    std::cout << result_vec.size() << std::endl;
//    draw_boxes(current_frame, result_vec);
//    cv::imshow(" ", current_frame);
//    cv::waitKey();
//#endif
//    return 0;
//}
//#include "src/parser.h"
//int main(void) {
//        std::string data_file = "/media/dat/05C830EB6380925C/data/plates-data/train";
//        std::string cfg_file = "/media/dat/05C830EB6380925C/data/plates-data/train/plate.cfg";
//        char *cfgfile = const_cast<char *>(cfg_file.data());
//    network net = parse_network_cfg_custom(cfgfile, 1, 0);
//}
#include "classifier_class.h"
int main(void)
{
    std::string cfg_file = "/media/dat/05C830EB6380925C/data/plates-data/train/plate.cfg";
    std::string weight_file = "/media/dat/05C830EB6380925C/data/plates-data/train/plate_last.weights";
    Classifier *cl = new Classifier(cfg_file, weight_file);
    cl->classify("/media/dat/05C830EB6380925C/data/plates-data/data/negative/neg_image_114105.jpg");
}
