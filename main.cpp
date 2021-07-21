
#include <memory>
#include <astra_core/astra_core.hpp>
#include <astra/astra.hpp>
#include <astra/capi/astra.h>
#include "LitDepthVisualizer.hpp"
#include <chrono>
#include <iostream>
#include <key_handler.h>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <vector>

#define MIN_H_BLUE 200
#define MAX_H_BLUE 300

using namespace std;
using namespace cv;

float elapsedMillis_{.0f};

using DurationType = std::chrono::milliseconds;
using ClockType = std::chrono::high_resolution_clock;

ClockType::time_point prev_;

using buffer_ptr = std::unique_ptr<int16_t[]>;
buffer_ptr buffer_;
unsigned int lastWidth_;
unsigned int lastHeight_;



astra::ColorStream configure_color(astra::StreamReader& reader)
{
auto colorStream = reader.stream<astra::ColorStream>();

auto oldMode = colorStream.mode();

//We don't have to set the mode to start the stream, but if you want to here is how:
astra::ImageStreamMode colorMode;

colorMode.set_width(640);
colorMode.set_height(480);
colorMode.set_pixel_format(astra_pixel_formats::ASTRA_PIXEL_FORMAT_RGB888);
colorMode.set_fps(30);

colorStream.set_mode(colorMode);

auto newMode = colorStream.mode();
printf("Changed color mode: %dx%d @ %d -> %dx%d @ %d\n",
    oldMode.width(), oldMode.height(), oldMode.fps(),
    newMode.width(), newMode.height(), newMode.fps());

return colorStream;
}

astra::DepthStream configure_depth(astra::StreamReader& reader)
{
auto depthStream = reader.stream<astra::DepthStream>();

auto oldMode = depthStream.mode();

//We don't have to set the mode to start the stream, but if you want to here is how:
astra::ImageStreamMode depthMode;

depthMode.set_width(640);
depthMode.set_height(480);
depthMode.set_pixel_format(astra_pixel_formats::ASTRA_PIXEL_FORMAT_DEPTH_MM);
depthMode.set_fps(30);

depthStream.set_mode(depthMode);

auto newMode = depthStream.mode();
printf("Changed depth mode: %dx%d @ %d -> %dx%d @ %d\n",
    oldMode.width(), oldMode.height(), oldMode.fps(),
    newMode.width(), newMode.height(), newMode.fps());

return depthStream;
}




int main(int argc, char** argv)
{
    // >>>> Kalman Filter
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;

    unsigned int type = CV_32F;
    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);

    cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
    cv::Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]
    //cv::Mat procNoise(stateSize, 1, type)
    // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    cv::setIdentity(kf.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
    // <<<< Kalman Filter
    double ticks = 0;
    bool found = false;
    int notFoundCount = 0;

    astra::initialize();

    set_key_handler();

    astra::StreamSet streamSet;
    astra::StreamReader reader = streamSet.create_reader();

    reader.stream<astra::ColorStream>().start();
    reader.stream<astra::DepthStream>().start();
    // may use for manual configuration of streams

    auto colorStream = configure_color(reader);
    auto depthStream = configure_depth(reader);
    
    vector<Point2f> colormat(4);
    vector<Point2f> depthmat(4);

    colormat[0] = Point2f (219, 133);
    colormat[1] = Point2f(489, 231);
    colormat[2] = Point2f(469, 87);
    colormat[3] = Point2f(184, 341);

    depthmat[0] = Point2f(216, 146);
    depthmat[1] = Point2f(475, 232);
    depthmat[2] = Point2f(453, 96);
    depthmat[3] = Point2f(191, 348);
    
    Mat trans = getPerspectiveTransform(depthmat, colormat);
    
    colorStream.start();
    depthStream.start();
    

    do
    {

        double precTick = ticks;
        ticks = (double)cv::getTickCount();

        double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

        astra::Frame frame = reader.get_latest_frame();
        const astra::ColorFrame colorFrame = frame.get<astra::ColorFrame>();

        const astra::DepthFrame depthFrame = frame.get<astra::DepthFrame>();

        cv::Mat mImageRGB(colorFrame.height(), colorFrame.width(), CV_8UC3, (void*)colorFrame.data());
        cv::Mat cImageBGR;
        cv::cvtColor(mImageRGB, cImageBGR, COLOR_RGB2BGR);
        Mat res;
        cImageBGR.copyTo(res);

        cv::Mat mImageDepth(depthFrame.height(), depthFrame.width(), CV_16UC1, (void*)depthFrame.data());
        cv::Mat mScaledDepth;
        mImageDepth.convertTo(mScaledDepth, CV_8U, 255.0 / 3500); 
        
        Mat transdepth;
        warpPerspective(mScaledDepth, transdepth, trans, Size(640, 480));
        //cv::imshow("transd", transd);
        

        //cv::imshow( "Color Image", cImageBGR ); // RGB image
        //cv::imshow( "Depth Image", mScaledDepth ); // depth image
        //cv::imshow( "gray",gray);
        //cv::imshow( "Depth Image 2", mImageDepth ); // contains depth data

        cv::Mat blur;
        cv::GaussianBlur(cImageBGR, blur, cv::Size(3, 3), 3.0, 3.0);
        cv::Mat frmHsv;
        cv::cvtColor(blur, frmHsv, COLOR_BGR2HSV);

        cv::Mat rangeRes = cv::Mat::zeros(cImageBGR.size(), CV_8UC1);
        cv::inRange(frmHsv, cv::Scalar(70, 100, 30),
            cv::Scalar(95, 250, 190), rangeRes);

        cv::erode(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        
        //cout << "value: " << int(frmHsv.at<Vec3b>(240, 320).val[0]) << " " << int(frmHsv.at<Vec3b>(240, 320).val[1]) << " " << int(frmHsv.at<Vec3b>(240, 320).val[2]) << endl;
        circle(rangeRes, Point(320, 240), 2, Scalar(255, 255, 255), 1);
        cv::imshow("Threshold", rangeRes);
        
        
        // >>>>> Contours detection
        vector<vector<cv::Point> > contours;
        cv::findContours(rangeRes, contours, RETR_EXTERNAL,
            CHAIN_APPROX_NONE);
        // <<<<< Contours detection

        // >>>>> Filtering
        vector<vector<cv::Point> > balls;
        vector<cv::Rect> ballsBox;
        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::Rect bBox;
            bBox = cv::boundingRect(contours[i]);
            

            // Searching for a bBox almost square
            if (bBox.area() >= 500)
            {
                balls.push_back(contours[i]);
                ballsBox.push_back(bBox);
            }
        }

        for (size_t i = 0; i < balls.size(); i++)
        {
            cv::rectangle(res, ballsBox[i], CV_RGB(0, 255, 0), 2);

            cv::Point center;
            center.x = ballsBox[i].x + ballsBox[i].width / 2;
            center.y = ballsBox[i].y + ballsBox[i].height / 2;


            stringstream sstr;
            sstr << "(" << center.x << "," << center.y << ")";
            cv::putText(res, sstr.str(),
                cv::Point(center.x + 3, center.y - 3),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20, 150, 20), 2);
        }



        imshow("contour", res);

        /*std::cout << "value: "    
        << mImageDepth.at<ushort>(240,320)
        << std::endl;*/
        
        if (cv::waitKey(1) == 'q')
            break;
        /*else if (cv::waitKey(1) == 'p')
        {
            cv::waitKey(0);
            Mat graycp;
            Mat depthcp;
            gray.copyTo(graycp); 
            mScaledDepth.copyTo(depthcp);

            vector<Vec3f> circlec;
            vector<Vec3f> circled;
            HoughCircles(graycp, circlec, HOUGH_GRADIENT, 1, 50, 100, 77, 0, 0);
            HoughCircles(depthcp, circled, HOUGH_GRADIENT, 1, 50, 100, 30, 0, 0);
            cout << "num: " << circlec.size() <<", "<<circled.size() << endl;
            if (circled.size() > 0) {
                int d1x = circled[0][0];
                int d1y = circled[0][1];
                int d1r = circled[0][2];

                cv::circle(depthcp, Point(d1x, d1y), d1r, Scalar(255, 255, 255), 1);

                cv::circle(depthcp, Point(d1x, d1y), 2, Scalar(255, 255, 255), 2);
                
                cv::imshow("cp2", depthcp);



                cout << "depth coord: " << d1x << "," << d1y << ", rad: " << d1r << endl;
                if (circlec.size() > 0) {
                    int c1x = circlec[0][0];
                    int c1y = circlec[0][1];
                    int c1r = circlec[0][2];
                    cv::circle(graycp, Point(c1x, c1y), c1r, Scalar(255, 255, 255), 1);
                    cv::circle(graycp, Point(c1x, c1y), 2, Scalar(0,0,0), 2);
                    cout << "color coord:" << c1x << ", " << c1y << ", rad: " << c1r << endl;
                    cv::imshow("cp1", graycp);
                }
            }

        }*/


        astra_update();



    } while (shouldContinue);

    astra::terminate();
}
