/*
	In Task 2, I use 'calcOpticalFlowFarneback' fuction to 
	calculate the optical flow of each pixel.
	The high value of flow means that the pixel is moving.
	In this way, we can find out moving object
*/

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "opencv2/bgsegm.hpp"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

void b_mode(char* videoFilename) 
{
	Mat frame;
	Mat fgMaskMOG2;
	char keyboard;
	
	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(100, 80, false); //MOG2 approach
//	pMOG2 = bgsegm::createBackgroundSubtractorGMG();
//	Ptr<BackgroundSubtractor> pMOG = bgsegm::createBackgroundSubtractorMOG();
	
	//create the capture object
	VideoCapture capture(videoFilename);
	if(!capture.isOpened()){
		//error in opening the video input
		cerr << "Unable to open video file: " << videoFilename << endl;
		exit(EXIT_FAILURE);
	}

	int  f = 0;
	keyboard = 0;
	while( keyboard != 'q' && keyboard != 27 ){
		//read the current frame
		if(!capture.read(frame)) {
			cerr << "End of video" << endl;
			exit(EXIT_FAILURE);
		}
		//update the background model
		pMOG2->apply(frame, fgMaskMOG2);
		
		// Store original fgMaskMOG2
		Size sz = frame.size();
		Mat ori_fgMaskMOG2(sz.height, sz.width, CV_8UC3);
		cvtColor(fgMaskMOG2, ori_fgMaskMOG2, CV_GRAY2RGB);
		
		// Remove nosie
//		Mat kernel_1 = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
//		morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, kernel_1);
		Mat kernel_2 = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
		morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_CLOSE, kernel_2);
	
		// Connect Components 
		Mat stats, centroids, labelImage;
		int nLabels = connectedComponentsWithStats(fgMaskMOG2, labelImage, stats, centroids, 8, CV_32S);
		Mat mask(labelImage.size(), CV_8UC1, Scalar(0));
		Mat surfSup = stats.col(CC_STAT_AREA) > 100;  // Size Threshold
		
		int car_count = 0;
		int person_count = 0;
		int other_count = 0;
		
		int object_count = 0 ;
		for (int i = 1; i < nLabels; i++)
		{
			if (surfSup.at<uchar>(i, 0))
			{
				// Naive classification by shape
				int width = stats.at<int>(i, CC_STAT_WIDTH);
				int height = stats.at<int>(i, CC_STAT_HEIGHT);
				
				if (height > width) {
					double result = abs(double(width)/double(height) - 0.3);
//					cout << "p: " << result << '\n';
					if ( result < 0.3) {
						person_count++;
					}
				} else {
					double result = abs(double(height)/double(width) - 0.7);
					cout << "c: " << result << '\n';
					if ( result < 0.4) {
						car_count++;
					}
				}	
//				cout << "width : " << width << '\n';
//				cout << "height: " << height << '\n';
				object_count++;
				mask = mask | (labelImage==i);
			}
		}
		
		
		// Set mask get moving object
		Mat objectImg(sz, CV_8UC1, Scalar(0));
		frame.copyTo(objectImg,mask);

		// Set mask get moving object
//		Mat objectImg(sz.height, sz.width, CV_8UC3);
//		objectImg.setTo(0);
//		frame.copyTo(objectImg, fgMaskMOG2);	
			
		// Get background
		Mat bgImg(sz.height, sz.width, CV_8UC3);
		pMOG2->getBackgroundImage(bgImg);
		
		// Make result frame
		Mat result(sz.height*2, sz.width*2, CV_8UC3);
		Mat leftTop = result(Rect(0, 0, sz.width, sz.height));
		frame.copyTo(leftTop);
		Mat rightTop = result(Rect(sz.width, 0, sz.width, sz.height));
		bgImg.copyTo(rightTop);
		Mat leftBot = result(Rect(0, sz.height, sz.width, sz.height));
		ori_fgMaskMOG2.copyTo(leftBot);
		Mat rightBot = result(Rect(sz.width, sz.height, sz.width, sz.height));
		objectImg.copyTo(rightBot);
		
		imshow("Result", result);
		cout << "Frame " << setw(4) << setfill('0') << f << ": "<< object_count << " objects ";
		f++;
		
		int all_count = car_count + person_count + other_count;
		other_count = all_count - car_count - person_count;
		

		if ( all_count > 0) {
			cout << " ( ";
			if (person_count > 0) {
				cout <<  person_count << " persons";
			}
			if (car_count > 0) {
				if (person_count > 0) { cout << ", "; }
				cout <<  car_count << " cars ";
			}
			
			if (other_count > 0) {
				if (car_count > 0) { cout << "and "; }
				cout <<  other_count << " others ";
			}
			
			cout << " ) ";
		}
		
		cout <<'\n';
		keyboard = (char)waitKey(30);
	} 
	//delete capture object
	capture.release();
}

/*---------------------------------------------------------------------------*/

#define UNKNOWN_FLOW_THRESH 1e1

void makecolorwheel(vector<Scalar> &colorwheel)
{
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    int i;
	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,	   255*i/RY,	 0));
    for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,		 0));
    for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,		   255,		 255*i/GC));
    for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,		   255-255*i/CB, 255));
    for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,	   0,		 255));
    for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,	   0,		 255-255*i/MR));
}
 
void motionToColor(Mat flow, Mat &color)
{
	if (color.empty())
		color.create(flow.rows, flow.cols, CV_8UC3);
 
	static vector<Scalar> colorwheel; //Scalar r,g,b
	if (colorwheel.empty())
		makecolorwheel(colorwheel);
 
	// determine motion range:
    float maxrad = -1;
 
	// Find max flow to normalize fx and fy
	for (int i= 0; i < flow.rows; ++i) 
	{
		for (int j = 0; j < flow.cols; ++j) 
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
				continue;
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = maxrad > rad ? maxrad : rad;
		}
	}
 
	for (int i= 0; i < flow.rows; ++i) 
	{
		for (int j = 0; j < flow.cols; ++j) 
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
 
			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			
			float rad = sqrt(fx * fx + fy * fy);
 
			float angle = atan2(-fy, -fx) / CV_PI;
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;
//			f = 0; // uncomment to see original color wheel
 
			for (int b = 0; b < 3; b++) 
			{
				float col0 =  
//				0.5;
				colorwheel[k0][b] / 255.0;
				float col1 = 
//				0.5;
				 colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius
				else
					col *= .75; // out of range
				data[2 - b] = (255 - (int)(255.0 * col));
//				data[2 - b] = ((int)(255.0 * col));
			}
		}
	}
}


void drawMotionField(Mat original, Mat flow) 
{
	for (int y = 0; y < original.rows; y += 5) 
	{
		for (int x = 0; x < original.cols; x += 5)
		{
			// get the flow from y, x position * 10 for better visibility
			const Point2f flowatxy = flow.at<Point2f>(y, x) * 10;
			// draw line at flow direction
			line(original, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255,0,0));
			// draw initial point
//			circle(original, Point(x, y), 1, Scalar(0, 0, 0), -1);
		}
	}
}

void s_mode(char* videoFilename) 
{
	char keyboard;
	Mat prevgray, gray, flow, cflow, frame;
	Mat motion2color;
	
	VideoCapture capture(videoFilename);
	if(!capture.isOpened()){
		//error in opening the video input
		cerr << "Unable to open video file: " << videoFilename << endl;
		exit(EXIT_FAILURE);
	}
	
	int  f = 0;
	keyboard = 0;
	while( keyboard != 'q' && keyboard != 27 ){
		keyboard = (char)waitKey(30);
		if(!capture.read(frame)) {
			cerr << "End of video" << endl;
			exit(EXIT_FAILURE);
		}
		
		Size sz = frame.size();
		
		cvtColor(frame, gray, CV_BGR2GRAY);
		
		if( prevgray.data )
		{
			calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
			motionToColor(flow, motion2color);
			
			Mat mf = frame.clone();
			drawMotionField(mf, flow);
			
			// Make result frame
			Mat result = Mat::zeros(sz.height*2, sz.width*2, CV_8UC3);
			Mat leftTop = result(Rect(0, 0, sz.width, sz.height));
			frame.copyTo(leftTop);
			Mat rightTop = result(Rect(sz.width, 0, sz.width, sz.height));
			mf.copyTo(rightTop);
			Mat leftBot = result(Rect(0, sz.height, sz.width, sz.height));
//			customChangeBG(motion2color);
			motion2color.copyTo(leftBot);
			
			Mat binary = motion2color.clone();
			cvtColor(motion2color, binary, CV_BGR2GRAY);
			threshold(binary, binary, 50,255,THRESH_BINARY);
			
			Mat kernel_2 = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
			morphologyEx(binary, binary, MORPH_CLOSE, kernel_2);
			
			// Connect Components 
			Mat stats, centroids, labelImage;
			int nLabels = connectedComponentsWithStats(binary, labelImage, stats, centroids, 8, CV_32S);
			Mat mask(labelImage.size(), CV_8UC1, Scalar(0));
			Mat surfSup = stats.col(CC_STAT_AREA) > 100;  // Size Threshold
			
			int object_count = 0 ;
			for (int i = 1; i < nLabels; i++)
			{
				if (surfSup.at<uchar>(i, 0))
				{
					mask = mask | (labelImage==i);
				}
			}
			
			// Set mask get moving object
			Mat objectImg(sz, CV_8UC1, Scalar(0));
			frame.copyTo(objectImg, mask);
			
//			cvtColor(binary, binary, CV_GRAY2RGB);
			Mat rightBot = result(Rect(sz.width, sz.height, sz.width, sz.height));
			objectImg.copyTo(rightBot);
			
			imshow("flow", result);
//			imshow("flow", gray);
		}
		std::swap(prevgray, gray);
		keyboard = (char)waitKey(30);
	} 
	//delete capture object
	capture.release();
	
}

int main(int argc, char* argv[])
{
	if (argc != 3) {
		printf("Invaild agrument!\n");
		return EXIT_FAILURE;
	}
	
	if (strcmp(argv[1], "-b") == 0) {
		b_mode(argv[2]);
	}
	
	if (strcmp(argv[1], "-s") == 0) {
		s_mode(argv[2]);
	}

	return EXIT_SUCCESS;
}