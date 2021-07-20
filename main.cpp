
#include <memory>
#include <astra_core/astra_core.hpp>
#include <astra/astra.hpp>
#include <astra/capi/astra.h>
#include "LitDepthVisualizer.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <key_handler.h>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"


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


void getcoord(int event, int x, int y, int flags, void* image)
{
    Mat* temp = reinterpret_cast<cv::Mat*>(image);
    if (event == EVENT_LBUTTONDOWN)
    {

        cout << "pixel: " << x << ", " << y << endl;

    }

}



int main(int argc, char** argv)
{
    
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


        astra::Frame frame = reader.get_latest_frame();
        const astra::ColorFrame colorFrame = frame.get<astra::ColorFrame>();

        const astra::DepthFrame depthFrame = frame.get<astra::DepthFrame>();

        cv::Mat mImageRGB(colorFrame.height(), colorFrame.width(), CV_8UC3, (void*)colorFrame.data());
        cv::Mat cImageBGR;
        cv::cvtColor(mImageRGB, cImageBGR, COLOR_RGB2BGR);
        cv::Mat gray;
        cv::cvtColor(mImageRGB, gray, COLOR_RGB2GRAY);
       

        cv::Mat mImageDepth(depthFrame.height(), depthFrame.width(), CV_16UC1, (void*)depthFrame.data());
        cv::Mat mScaledDepth;

        mImageDepth.convertTo(mScaledDepth, CV_8U, 255.0 / 3500); 
        
        Mat transd;
        warpPerspective(mScaledDepth, transd, trans, Size(640, 480));
        cv::imshow("transd", transd);
        

        cv::imshow( "Color Image", cImageBGR ); // RGB image
        cv::imshow( "Depth Image", mScaledDepth ); // depth image
        cv::imshow( "gray",gray);
        //cv::imshow( "Depth Image 2", mImageDepth ); // contains depth data



        cv::Mat result;
        cv::addWeighted(gray, 0.5, transd, 0.5, 0.0,result);
        cv::imshow("overlayl", result);
        cv::Mat result2;
        cv::addWeighted(gray, 0.5, mScaledDepth, 0.5, 0.0, result2);
        cv::imshow("overlayf", result2);

        /*std::cout << "value: "
        << mImageDepth.at<ushort>(240,320)
        << std::endl;*/
        

        /*std::cout << "colorStream -- hFov: "
        << reader.stream<astra::ColorStream>().hFov()
        << " vFov: "
        << reader.stream<astra::ColorStream>().vFov()
        << std::endl;

        std::cout << "depthStream -- hFov: "
        << reader.stream<astra::DepthStream>().hFov()
        << " vFov: "
        << reader.stream<astra::DepthStream>().vFov()
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
