
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
#include "opencv2/features2d.hpp"
#include "opencv2/video/background_segm.hpp"
#include <vector>
#include <numeric>
#include <algorithm>

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



int polyRegression(const deque<Point2f>& center, const int yy) {
    int n = center.size();
    vector<float> x(n);
    vector<float> y(n);
    for (int i = 0; i < n; i++)
    {
        x[i] = center[i].x;
        y[i] = center[i].y;

    }
    float xm = std::accumulate(x.begin(), x.end(), 0.0) / n;
    float ym = std::accumulate(y.begin(), y.end(), 0.0) / n;
    std::vector<float> x2(n);
    for (int i = 0; i < n; i++)
    {
        x2[i] = pow(x[i], 2);
    }
    float x2m = std::accumulate(x2.begin(), x2.end(), 0.0) / n;
    float x3m = std::inner_product(x2.begin(), x2.end(), x.begin(), 0.0) / n;
    float x4m = std::inner_product(x2.begin(), x2.end(), x2.begin(), 0.0) / n;

    float xym = std::inner_product(x.begin(), x.end(), y.begin(), 0.0) / n;
    float x2ym = std::inner_product(x2.begin(), x2.end(), y.begin(), 0.0) / n;

    float sxx = x2m - xm * xm;
    float sxy = xym - xm * ym;
    float sxx2 = x3m - xm * x2m;
    float sx2x2 = x4m - x2m * x2m;
    float sx2y = x2ym - x2m * ym;

    float b = (sxy * sx2x2 - sx2y * sxx2) / (sxx * sx2x2 - sxx2 * sxx2);
    float c = (sx2y * sxx - sxy * sxx2) / (sxx * sx2x2 - sxx2 * sxx2);
    float a = ym - b * xm - c * x2m;

 
    return (sqrt(b * b - 4 * c * (a - yy)) - b) / (2 * c);
   
    
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


    Ptr<BackgroundSubtractor> pmog2;
    pmog2 = createBackgroundSubtractorMOG2(500,25.0,false);
    
    SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.minArea = 800;
    params.filterByCircularity = true;
    params.minCircularity = 0.35;
    params.filterByConvexity = true;
    params.minConvexity = 0.8;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    colorStream.start();
    depthStream.start();
    
    deque<Point2f> center;
    deque<Point2f> centerz;
    
    int count = 0;
    int yy = -1;
    int xx = -1;
    int zz = -1;

    do
    {

        astra::Frame frame = reader.get_latest_frame();
        const astra::ColorFrame colorFrame = frame.get<astra::ColorFrame>();

        const astra::DepthFrame depthFrame = frame.get<astra::DepthFrame>();

        cv::Mat mImageRGB(colorFrame.height(), colorFrame.width(), CV_8UC3, (void*)colorFrame.data());
        cv::Mat cImageBGR;
        cv::cvtColor(mImageRGB, cImageBGR, COLOR_RGB2BGR);
        cv::Mat mImageDepth(depthFrame.height(), depthFrame.width(), CV_16UC1, (void*)depthFrame.data());
        cv::Mat mScaledDepth;
        mImageDepth.convertTo(mScaledDepth, CV_8U, 255.0 / 3500);

        Mat transdepth;
        warpPerspective(mImageDepth, transdepth, trans, Size(640, 480));
        Mat transcale;
        warpPerspective(mScaledDepth, transcale, trans, Size(640, 480));


        // BLOB DETECTION
        Mat mask;
        Mat maski;
        pmog2->apply(cImageBGR, mask);
        erode(mask, mask, Mat(), Point(-1, -1), 1);
        dilate(mask, mask, Mat(), Point(-1, -1), 1);
        bitwise_not(mask,maski);

        vector<KeyPoint> keypoints;
        detector->detect(maski, keypoints);
        Mat blob;
        Mat transcp;
        drawKeypoints(cImageBGR, keypoints, blob, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(transcale, keypoints, transcp, Scalar(255, 255,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        //cout << transdepth.at<ushort>(420, 520) << endl;
        //circle(transcp, Point(520, 420), 2, (255, 255, 0), 2);
        
        //여기부터 자세히 봐주셈
        if (keypoints.size() > 0) {
            
            
            center.push_front(keypoints[0].pt);
            centerz.push_front(Point2f((transdepth.at<ushort>(keypoints[0].pt.y, keypoints[0].pt.x)),keypoints[0].pt.y ));
            //cout <<"coord: " <<centerz.begin()[0]  << endl;
            if (keypoints[0].pt.x >440 && count == 0)
            {
                cout << "predict starting at: " << center.begin()[0] << ", " << transdepth.at<ushort>(keypoints[0].pt.y, keypoints[0].pt.x) << endl;
                for (int i = 0; i < (430 - (keypoints[0].pt.y+40)) / 7; i++)
                {
                    int yp = keypoints[0].pt.y+40 + 7 * i;
                    int zp = polyRegression(centerz, yp);
                    int xp = polyRegression(center, yp);

                    if (xp<640 && abs((transdepth.at<ushort>(yp,xp)-zp))<300)
                    {
                        xx = xp;
                        yy = yp;
                        zz = zp;
                        
                    }
                }
               
               cout << "coord prediction: " << xx << ", " << yy << ", " << zz << endl;
               count++;
            }
        }
        else if ((keypoints.size() == 0 || center.size() > 7) && center.size() > 0)
        {
            center.pop_back();
            centerz.pop_back();
        }
        
        if (center.size() > 5) {

            for (int i = 1; i < 5; i++)
            {
                line(blob, Point2i(center[i - 1]), Point2i(center[i]), Scalar(0, 255, 0), 2);
            }
        }
        if (xx != -1) { circle(blob, Point(xx, yy), 2, Scalar(255, 255, 0), 2); }
        //여기까지
        
        

        //cv::imshow( "Color Image", cImageBGR ); // RGB image
        //cv::imshow( "Depth Image", mScaledDepth ); // depth image
        //cv::imshow( "gray",gray);
        //cv::imshow( "Depth Image 2", mImageDepth ); // contains depth data
        imshow("mask", mask);
        imshow("blob", blob);
        //imshow("depthblob", transcp);
        //imshow("mask", mask);

        
        
        if (cv::waitKey(1) == 'q')
            break;
        else if (cv::waitKey(1) == 'c')
        {
            
            if (count == 1) { count--; 
            xx = -1; yy = -1; zz = -1;
            }
            cout << count << endl;
        }


        astra_update();



    } while (shouldContinue);

    astra::terminate();
}
