/******************************************************************

Rescale input image rule:
	- Single input image, resize as Size(600,480)
	- Multiple input images, resize the height as 600 and the width = 600 * orignal_ratio
 lo
******************************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
const int draw_shift_bits = 4;
const int draw_multiplier = 1 << draw_shift_bits;

Mat open_img_resize(char filename[], bool as_ratio)
{
	Mat img = imread(filename, CV_LOAD_IMAGE_UNCHANGED);

	if(img.empty())
	{
		fprintf(stderr, "failed to load input image\n");
	}
	
	Size dsize = Size(600,480);
	if (as_ratio) {
		Size sz = img.size();
		double ratio = sz.width/double(sz.height);
		Size dsize1 = Size(ratio*600, 600);
		Mat img2 = Mat(dsize1,CV_32S);
		resize(img, img2, dsize1);
		return img2;
	}
	Mat img2 = Mat(dsize,CV_32S);
	resize(img, img2, dsize);
	return img2;
}

std::vector<cv::KeyPoint> get_keypoints(Mat img)
{
	cv::SiftFeatureDetector detector;
	std::vector<cv::KeyPoint> keypoints;
	detector.detect(img, keypoints);
	return keypoints;
}

Mat get_descriptors(Mat img, std::vector<cv::KeyPoint> keypoints)
{
	SiftDescriptorExtractor extractor;
	Mat descriptors;
	extractor.compute(img, keypoints, descriptors);
	return descriptors;
}

void drawCustomKeyPoints(Mat img, const cv::KeyPoint &p, const Scalar &color)
{
	// cv ::'LINE_AA' = 16
	Point center( cvRound(p.pt.x * draw_multiplier), cvRound(p.pt.y * draw_multiplier) );
	int radius = cvRound(p.size/2 * draw_multiplier);
	circle(img, center, radius, color, 1, 16, draw_shift_bits );
	Point piont(p.pt.x, p.pt.y);
	int size_r = cvRound(p.size/40 * draw_multiplier);
	drawMarker(img, piont, color,  cv::MARKER_CROSS, size_r, 1, 16);
}

Mat custom_draw_points(Mat image, std::vector<cv::KeyPoint> keypoints)
{
	Mat outImage;
	if( image.type() == CV_8UC3 )
	{
		image.copyTo( outImage );
	} else if( image.type() == CV_8UC1 )
	{
		cvtColor( image, outImage, COLOR_GRAY2BGR );
	}else{
		printf("Incorrect type of input image.\n");
	}
	
	RNG& rng=theRNG();
	std::vector<KeyPoint>::const_iterator it = keypoints.begin(), end = keypoints.end();
	for( ; it != end; ++it )
	{
		Scalar color = Scalar(rng(256), rng(256), rng(256));
		drawCustomKeyPoints(outImage, *it, color);
	}
	return outImage;
}

void single_img(char *imgName)
{
	Mat img1 = open_img_resize(imgName, false);
	std::vector<cv::KeyPoint> keypoints1 = get_keypoints(img1);
	printf("# of keypoints in %s is %li\n", imgName, keypoints1.size());
	cv::Mat output1 = custom_draw_points(img1, keypoints1);
	Size sz = img1.size();
//	Size sz2 = output1.size();
	Mat img3(sz.height, sz.width*2, CV_8UC3);
	Mat left(img3, Rect(0, 0, sz.width, sz.height));
	img1.copyTo(left);
	Mat right(img3, Rect(sz.width, 0, sz.width, sz.height));
	output1.copyTo(right);
	imshow("Display Image", img3);
	waitKey(0);
}

void multiple_images(int K_percentage, int argc, char** argv)
{
	printf("\n");
	
	Mat featuresUnclustered;
	Mat histograms;
	int total_num_keypoints = 0;
	
	int i;
	for (i=1;i<argc;i++) {
		char *imgName = argv[i];
		Mat img = open_img_resize(imgName, true);
		std::vector<cv::KeyPoint> keypoints = get_keypoints(img);
		total_num_keypoints += keypoints.size();
		Mat descriptors = get_descriptors(img, keypoints);
		featuresUnclustered.push_back(descriptors);
	}
	
	int feature_num = (K_percentage / double(100))*total_num_keypoints;
	printf("K = %d%% * (total number of keypoionts) = %d\n\n", K_percentage, feature_num);
	BOWKMeansTrainer bowTraining(feature_num);            
	Mat dictionary = bowTraining.cluster(featuresUnclustered);
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("SIFT"); 
	cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("FlannBased");
	cv::BOWImgDescriptorExtractor bowDE(extractor, matcher);  
	bowDE.setVocabulary(dictionary);
	
	for (i=1;i<argc;i++) {
		char *imgName = argv[i];
		Mat img = open_img_resize(imgName, true);
		std::vector<cv::KeyPoint> keypoints = get_keypoints(img);
		Mat descriptors = featuresUnclustered.row(i-1);
		bowDE.compute(img, keypoints, descriptors); 
		histograms.push_back(descriptors);
	}

	char s[] = " ";
	printf("%-*s", 15, s);
	for (i=1;i<argc;i++) {
		char *imgName = argv[i];
		printf("%-*s", 15, imgName);
	}
	printf("\n");
		
	for (i=1;i<argc;i++) {
		char *imgName = argv[i];
		printf("%-*s", 15, imgName);
		Mat img = open_img_resize(imgName, true);
		std::vector<cv::KeyPoint> keypoints = get_keypoints(img);
		Mat descriptors = featuresUnclustered.row(i-1);
		bowDE.compute(img, keypoints, descriptors);
		
		for (int j=0;j<argc-1;j++) {
			double dist = compareHist(descriptors, histograms.row(j),CV_COMP_CHISQR); // dissimilarity
			printf("%-*.2f", 15, dist);
		}
		printf("\n");
	}
}

int main(int argc, char** argv)
{   
	if (argc == 2) {
		single_img(argv[1]);
		return 0;
	}
	
//	Mat all_imgs;
	int i;
	for (i=1;i<argc;i++) {
		char *imgName = argv[i];
		Mat img = open_img_resize(imgName, true);
//		all_imgs.push_back(img);
		std::vector<cv::KeyPoint> keypoints = get_keypoints(img);
		printf("# of keypoints in %s is %li\n", imgName, keypoints.size());
	}
	
	multiple_images(5, argc, argv);
	multiple_images(10, argc, argv);
	multiple_images(20, argc, argv);
	
	return 0;
}