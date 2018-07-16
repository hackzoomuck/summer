//-----watershed
//#include <iostream>
//
//#include <opencv2/highgui/highgui.hpp>
//
//#include <opencv2/core/core.hpp>
//
//#include <opencv2/imgproc/imgproc.hpp>
//
//#include "watershedSegment.h"
//
//
//int main() {
//
//	cv::Mat image = cv::imread("city5.jpg");
//
//	cv::imshow("Original Image", image); //원본
//
//	cv::Mat gray_image;
//
//	cv::cvtColor(image, gray_image, CV_BGR2GRAY);
//
//	cv::imshow("Gray Image", gray_image); //gray영상
//
//	cv::Mat binary_image;
//
//	cv::threshold(gray_image, binary_image, 90, 255, cv::THRESH_BINARY_INV);
//
//	cv::imshow("Binary Image", binary_image); //이진영상으로변환(손하얗게끔inverse)
//
//
//
//	cv::Mat fg;
//
//	cv::erode(binary_image, fg, cv::Mat(), cv::Point(-1, -1), 12); //침식
//
//	cv::imshow("Foreground", fg);
//
//
//
//	cv::Mat bg;
//
//	cv::dilate(binary_image, bg, cv::Mat(), cv::Point(-1, -1), 40); //팽창
//
//	cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);
//
//	//(1보다작은)배경을128, (1보다큰)객체0. Threshold설정INVERSE 적용.
//
//	cv::imshow("Background", bg);
//
//
//
//	cv::Mat markers(binary_image.size(), CV_8U, cv::Scalar(0));
//
//	markers = fg + bg; //침식+팽창= 마커영상으로조합. 워터쉐드알고리즘에 입력으로 사용됨.
//
//	cv::imshow("Marker", markers);
//
//
//
//	WatershedSegmenter segmenter; //워터쉐드분할객체생성
//
//	segmenter.setMarkers(markers); //set마커하면signed 이미지로바뀜
//
//	segmenter.process(image); //0,128,255로구성됨
//
//	cv::imshow("Segmentation", segmenter.getSegmentation());
//
//
//
//	cv::imshow("Watershed", segmenter.getWatersheds()); // 0,255로구성됨
//
//	cv::waitKey(0);
//
//	return 0;
//
//}
////------grabcut
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//#include <iostream>
//using namespace std;
//using namespace cv;
//static void help()
//{
//	cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
//		"and then grabcut will attempt to segment it out.\n"
//		"Call:\n"
//		"./grabcut <image_name>\n"
//		"\nSelect a rectangular area around the object you want to segment\n" <<
//		"\nHot keys: \n"
//		"\tESC - quit the program\n"
//		"\tr - restore the original image\n"
//		"\tn - next iteration\n"
//		"\n"
//		"\tleft mouse button - set rectangle\n"
//		"\n"
//		"\tCTRL+left mouse button - set GC_BGD pixels\n"
//		"\tSHIFT+left mouse button - set GC_FGD pixels\n"
//		"\n"
//		"\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
//		"\tSHIFT+right mouse button - set GC_PR_FGD pixels\n" << endl;
//}
//const Scalar RED = Scalar(0, 0, 255);
//const Scalar PINK = Scalar(230, 130, 255);
//const Scalar BLUE = Scalar(255, 0, 0);
//const Scalar LIGHTBLUE = Scalar(255, 255, 160);
//const Scalar GREEN = Scalar(0, 255, 0);
//const int BGD_KEY = EVENT_FLAG_CTRLKEY;
//const int FGD_KEY = EVENT_FLAG_SHIFTKEY;
//static void getBinMask(const Mat& comMask, Mat& binMask)
//{
//	if (comMask.empty() || comMask.type() != CV_8UC1)
//		CV_Error(Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
//	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
//		binMask.create(comMask.size(), CV_8UC1);
//	binMask = comMask & 1;
//}
//class GCApplication
//{
//public:
//	enum { NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
//	static const int radius = 2;
//	static const int thickness = -1;
//	void reset();
//	void setImageAndWinName(const Mat& _image, const string& _winName);
//	void showImage() const;
//	void mouseClick(int event, int x, int y, int flags, void* param);
//	int nextIter();
//	int getIterCount() const { return iterCount; }
//private:
//	void setRectInMask();
//	void setLblsInMask(int flags, Point p, bool isPr);
//	const string* winName;
//	const Mat* image;
//	Mat mask;
//	Mat bgdModel, fgdModel;
//	uchar rectState, lblsState, prLblsState;
//	bool isInitialized;
//	Rect rect;
//	vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
//	int iterCount;
//};
//void GCApplication::reset()
//{
//	if (!mask.empty())
//		mask.setTo(Scalar::all(GC_BGD));
//	bgdPxls.clear(); fgdPxls.clear();
//	prBgdPxls.clear();  prFgdPxls.clear();
//	isInitialized = false;
//	rectState = NOT_SET;
//	lblsState = NOT_SET;
//	prLblsState = NOT_SET;
//	iterCount = 0;
//}
//void GCApplication::setImageAndWinName(const Mat& _image, const string& _winName)
//{
//	if (_image.empty() || _winName.empty())
//		return;
//	image = &_image;
//	winName = &_winName;
//	mask.create(image->size(), CV_8UC1);
//	reset();
//}
//void GCApplication::showImage() const
//{
//	if (image->empty() || winName->empty())
//		return;
//	Mat res;
//	Mat binMask;
//	if (!isInitialized)
//		image->copyTo(res);
//	else
//	{
//		getBinMask(mask, binMask);
//		image->copyTo(res, binMask);
//	}
//	vector<Point>::const_iterator it;
//	for (it = bgdPxls.begin(); it != bgdPxls.end(); ++it)
//		circle(res, *it, radius, BLUE, thickness);
//	for (it = fgdPxls.begin(); it != fgdPxls.end(); ++it)
//		circle(res, *it, radius, RED, thickness);
//	for (it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it)
//		circle(res, *it, radius, LIGHTBLUE, thickness);
//	for (it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it)
//		circle(res, *it, radius, PINK, thickness);
//	if (rectState == IN_PROCESS || rectState == SET)
//		rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), GREEN, 2);
//	imshow(*winName, res);
//}
//void GCApplication::setRectInMask()
//{
//	CV_Assert(!mask.empty());
//	mask.setTo(GC_BGD);
//	rect.x = max(0, rect.x);
//	rect.y = max(0, rect.y);
//	rect.width = min(rect.width, image->cols - rect.x);
//	rect.height = min(rect.height, image->rows - rect.y);
//	(mask(rect)).setTo(Scalar(GC_PR_FGD));
//}
//void GCApplication::setLblsInMask(int flags, Point p, bool isPr)
//{
//	vector<Point> *bpxls, *fpxls;
//	uchar bvalue, fvalue;
//	if (!isPr)
//	{
//		bpxls = &bgdPxls;
//		fpxls = &fgdPxls;
//		bvalue = GC_BGD;
//		fvalue = GC_FGD;
//	}
//	else
//	{
//		bpxls = &prBgdPxls;
//		fpxls = &prFgdPxls;
//		bvalue = GC_PR_BGD;
//		fvalue = GC_PR_FGD;
//	}
//	if (flags & BGD_KEY)
//	{
//		bpxls->push_back(p);
//		circle(mask, p, radius, bvalue, thickness);
//	}
//	if (flags & FGD_KEY)
//	{
//		fpxls->push_back(p);
//		circle(mask, p, radius, fvalue, thickness);
//	}
//}
//void GCApplication::mouseClick(int event, int x, int y, int flags, void*)
//{
//	// TODO add bad args check
//	switch (event)
//	{
//	case EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
//	{
//		bool isb = (flags & BGD_KEY) != 0,
//			isf = (flags & FGD_KEY) != 0;
//		if (rectState == NOT_SET && !isb && !isf)
//		{
//			rectState = IN_PROCESS;
//			rect = Rect(x, y, 1, 1);
//		}
//		if ((isb || isf) && rectState == SET)
//			lblsState = IN_PROCESS;
//	}
//	break;
//	case EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
//	{
//		bool isb = (flags & BGD_KEY) != 0,
//			isf = (flags & FGD_KEY) != 0;
//		if ((isb || isf) && rectState == SET)
//			prLblsState = IN_PROCESS;
//	}
//	break;
//	case EVENT_LBUTTONUP:
//		if (rectState == IN_PROCESS)
//		{
//			rect = Rect(Point(rect.x, rect.y), Point(x, y));
//			rectState = SET;
//			setRectInMask();
//			CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
//			showImage();
//		}
//		if (lblsState == IN_PROCESS)
//		{
//			setLblsInMask(flags, Point(x, y), false);
//			lblsState = SET;
//			showImage();
//		}
//		break;
//	case EVENT_RBUTTONUP:
//		if (prLblsState == IN_PROCESS)
//		{
//			setLblsInMask(flags, Point(x, y), true);
//			prLblsState = SET;
//			showImage();
//		}
//		break;
//	case EVENT_MOUSEMOVE:
//		if (rectState == IN_PROCESS)
//		{
//			rect = Rect(Point(rect.x, rect.y), Point(x, y));
//			CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
//			showImage();
//		}
//		else if (lblsState == IN_PROCESS)
//		{
//			setLblsInMask(flags, Point(x, y), false);
//			showImage();
//		}
//		else if (prLblsState == IN_PROCESS)
//		{
//			setLblsInMask(flags, Point(x, y), true);
//			showImage();
//		}
//		break;
//	}
//}
//int GCApplication::nextIter()
//{
//	if (isInitialized)
//		grabCut(*image, mask, rect, bgdModel, fgdModel, 1);
//	else
//	{
//		if (rectState != SET)
//			return iterCount;
//		if (lblsState == SET || prLblsState == SET)
//			grabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK);
//		else
//			grabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
//		isInitialized = true;
//	}
//	iterCount++;
//	bgdPxls.clear(); fgdPxls.clear();
//	prBgdPxls.clear(); prFgdPxls.clear();
//	return iterCount;
//}
//GCApplication gcapp;
//static void on_mouse(int event, int x, int y, int flags, void* param)
//{
//	gcapp.mouseClick(event, x, y, flags, param);
//}
//int main(int argc, char** argv)
//{
//	cv::CommandLineParser parser(argc, argv, "{@input| ../data/messi5.jpg |}");
//	help();
//	string filename = parser.get<string>("@input");
//	if (filename.empty())
//	{
//		cout << "\nDurn, empty filename" << endl;
//		return 1;
//	}
//	Mat image = imread("city.jpg");
//	
//	const string winName = "image";
//	namedWindow(winName, WINDOW_AUTOSIZE);
//	setMouseCallback(winName, on_mouse, 0);
//	gcapp.setImageAndWinName(image, winName);
//	gcapp.showImage();
//	for (;;)
//	{
//		char c = (char)waitKey(0);
//		switch (c)
//		{
//		case '\x1b':
//			cout << "Exiting ..." << endl;
//			goto exit_main;
//		case 'r':
//			cout << endl;
//			gcapp.reset();
//			gcapp.showImage();
//			break;
//		case 'n':
//			int iterCount = gcapp.getIterCount();
//			cout << "<" << iterCount << "... ";
//			int newIterCount = gcapp.nextIter();
//			if (newIterCount > iterCount)
//			{
//				gcapp.showImage();
//				cout << iterCount << ">" << endl;
//			}
//			else
//				cout << "rect must be determined>" << endl;
//			break;
//		}
//	}
//exit_main:
//	destroyWindow(winName);
//	return 0;
//}

////-----이미지 합성-----
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <opencv/highgui.h>
//#include <opencv/cv.h>
//#include <stdio.h>
//#include <stdlib.h>
//
//
//void main()
//
//{
//
//	IplImage *sum_image = 0, *image1 = 0, *image2 = 0;
//
//
//
//	image1 = cvLoadImage("city.jpg", -1);
//
//	image2 = cvLoadImage("city7.jpg", -1);
//
//
//
//	sum_image = cvCreateImage(cvGetSize(image2), 8, 3);
//
//	cvAdd(image1, image2, sum_image, NULL);
//
//
//
//	cvNamedWindow("Image 1", CV_WINDOW_AUTOSIZE);
//
//	cvNamedWindow("Image 2", CV_WINDOW_AUTOSIZE);
//
//	cvNamedWindow("Sum Image", CV_WINDOW_AUTOSIZE);
//
//	cvShowImage("Image 1", image1);
//
//	cvShowImage("Image 2", image2);
//
//	cvShowImage("Sum Image", sum_image);
//
//
//
//	cvWaitKey(0);
//}
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#define THRESHOLD 100
#define BRIGHT 0.7
#define DARK 0.2


using namespace cv;
using namespace std;

Mat canny_dst;
Mat canny_rem;
Mat area1, area2;
int thresh;
int check, check_back;
Mat src; Mat src_gray;
int max_thresh = 255;
RNG rng(12345);
int ans;
int entire;
void test();
void thresh_callback(int, void*)
{
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	const char* src_name = "Source";
	const char* canny_name = "Canny";
	const char* canny_name2 = "Canny2";


	if (!(check_back == 0 || check == 1)) {
		src_gray = canny_rem;
	}


	Canny(src_gray, canny_output, thresh, thresh * 2, 3);

	/// Find contours 
	findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw contours 
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(255, 255, 255);
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point2i());
		//   cout << "# of contour points: " << contours[i].size() << std::endl;

		for (unsigned int j = 0; j < contours[i].size(); j++)
		{
			ans += contourArea(contours[i]);
			//cout << "Point(x,y)=" << contours[i][j] << std::endl;
		}
	}


	/// Show in a window 

	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);

	//   findContours(canny_output, canny_dst, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	canny_dst = drawing.clone();

	if (check_back == 0 && check != 1) {
		imshow(canny_name2, canny_dst);
		area2 = canny_dst;
	}
	else if (check == 1) {
		imshow(canny_name, canny_dst);
		area1 = canny_dst;
	}
	else {
		imshow(canny_name, canny_dst);
		area1 = canny_dst;
	}


	check++;
}


class Histogram1D {
private:
	int histSize[1]; // 빈도수
	float hranges[2]; // 최소/최대 화소값
	const float* ranges[1];
	int channels[1]; // 여기서 1채널만 사용
public:
	Histogram1D() { // 1차원 히스토그램을 위한 인자 준비
		histSize[0] = 256;
		hranges[0] = 0.0;
		hranges[1] = 255.0;
		ranges[0] = hranges;
		channels[0] = 0; // 기본적으로 채널을 0으로 보기
	}

	// 정의한 멤버 변수로 그레이레벨 영상의 히스토그램을 계산할 때는 다음 메소드를 사용해 수행
	cv::MatND getHistogram(const cv::Mat &image) {
		// 1차원(1D) 히스토그램 계산.
		cv::MatND hist;
		cv::calcHist(&image, // 히스토그램 계산 
			1,   // 단일 영상의 히스토그램만
			channels, // 대상 채널               
			cv::Mat(), // 마스크 사용하지 않음     
			hist,  // 결과 히스토그램         
			1,   // 1차원(1D) 히스토그램           
			histSize, // 빈도수                  
			ranges  // 화소값 범위             
		);
		return hist;
	}

	// 값의 순서만으로 의미를 파악하기 어려우므로 바 그래프를 사용
	// 그래프를 생성하는 메소드
	cv::Mat getHistogramImage(const cv::Mat &image) {
		// 1차원(1D) 히스토그램을 계산하고, 영상으로 반환

		cv::MatND hist = getHistogram(image); // 먼저 히스토그램 계산

		double maxVal = 0; // 최대 빈도수 가져오기
		double minVal = 0; // 최소 빈도수 가져오기
		cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

		cv::Mat histImg(histSize[0], histSize[0], CV_8U, cv::Scalar(255));
		// 히스토그램을 출력하기 위한 영상

		int hpt = static_cast<int>(0.9*histSize[0]);
		// nbins의 90%를 최대점으로 설정


		for (int h = 0; h < histSize[0]; h++) { // 각 빈도에 대한 수직선을 그리기 
			float binVal = hist.at<float>(h);
			int intensity = static_cast<int>(binVal*hpt / maxVal);
			cv::line(histImg, cv::Point(h, histSize[0]), cv::Point(h, histSize[0] - intensity), cv::Scalar::all(0));
			// 두 점 간의 거리를 그리는 함수
		}
		return histImg;
	}
};


void hst() {
	Histogram1D h1; // 히스토그램 객체
	cv::MatND histo1 = h1.getHistogram(area1); // 히스토그램 계산

	float sum1 = 0;
	for (int i = 0; i < 256; i++) { // 각 빈도 조회
									//   std::cout << "Value " << i << " = " << histo.at<float>(i) << std::endl;
		sum1 += histo1.at<float>(i);
		//cout << i << " is " << histo1.at<float>(i)<<endl;
	}
	//   sum1 -= histo1.at<float>(0);
	Histogram1D h2; // 히스토그램 객체
	cv::MatND histo2 = h2.getHistogram(area2); // 히스토그램 계산

	float sum2 = 0;
	for (int i = 0; i < 256; i++) { // 각 빈도 조회
									//   std::cout << "Value " << i << " = " << histo.at<float>(i) << std::endl;
		sum2 += histo2.at<float>(i);

		//cout << i << " is " << histo1.at<float>(i) << endl;
	}
	//   sum2 -= histo2.at<float>(0);

	//cout << "다리영역 sum1 << "  " << sum2<<endl;
	float sum3 = sum2 - sum1;
	cout << "하지정맥 영역: " << sum3 << endl;
	cout << "하지정맥이 차지하는 퍼센트" << (sum3 / entire) * 100 << endl;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN) {
		hst();
	}
	if (event == EVENT_RBUTTONDOWN) {
		cout << "왼쪽 마우스 버튼 클릭.. 좌표 = (" << x << ", " << y << ")" << endl;
	}
	if (event == EVENT_MOUSEWHEEL) {
		if (getMouseWheelDelta(flags)>0)
			thresh += 5;
		else
			thresh -= 5;

		check_back = 1;
		printf("thresh : %d\n", thresh);
		test();

	}
}

void test() {
	check = 1;

	Mat inputImg, resultImg;
	Mat hlsImg, rgbImg;
	Mat skinImg, veinImg;
	Mat img_gray;
	inputImg = imread("city.jpg");
	CV_Assert(inputImg.depth() == CV_8U);
	resultImg.create(inputImg.size(), inputImg.type());

	const int nChannels = inputImg.channels();

	for (int j = 1; j < inputImg.rows - 1; ++j)
	{
		const uchar* prev = inputImg.ptr(j - 1);
		const uchar* curr = inputImg.ptr(j);
		const uchar* next = inputImg.ptr(j + 1);
		uchar* output = resultImg.ptr(j);
		for (int i = nChannels; i < nChannels*(inputImg.cols - 1); ++i)
		{
			*output++ = saturate_cast<uchar>(5 * curr[i] - curr[i - nChannels] - curr[i + nChannels] - prev[i] - next[i]);
		}
	}

	resultImg.row(0).setTo(Scalar(0));
	resultImg.row(resultImg.rows - 1).setTo(Scalar(0));
	resultImg.col(0).setTo(Scalar(0));
	resultImg.col(resultImg.cols - 1).setTo(Scalar(0));


	/// Load source image and convert it to gray
	src = resultImg;

	//resize(inputImg, inputImg, Size(), 0.4, 0.4, CV_INTER_AREA);
	skinImg = src.clone();

	cvtColor(skinImg, hlsImg, CV_BGR2HLS);
	vector<Mat> hls_images(3);
	split(hlsImg, hls_images);

	for (int row = 0; row < hlsImg.rows; row++)
	{
		for (int col = 0; col < hlsImg.cols; col++)
		{
			uchar H = hlsImg.at<Vec3b>(row, col)[0];
			uchar L = hlsImg.at<Vec3b>(row, col)[1];
			uchar S = hlsImg.at<Vec3b>(row, col)[2];

			double LS_ratio = ((double)L) / ((double)S);
			bool skin_pixel = (S >= 0.2) && (LS_ratio > 0.5) && (LS_ratio < 30.0) && ((H <= 50) || (H >= 165));

			if (skin_pixel == false)
			{
				skinImg.at<Vec3b>(row, col)[0] = 0;
				skinImg.at<Vec3b>(row, col)[1] = 0;
				skinImg.at<Vec3b>(row, col)[2] = 0;
			}
		}
	}

	namedWindow("Original", CV_WINDOW_AUTOSIZE);
	namedWindow("SkinDetected", CV_WINDOW_AUTOSIZE);


	moveWindow("Original", 100, 100);
	moveWindow("SkinDetected", 120, 120);


	imshow("Original", inputImg);
	imshow("SkinDetected", skinImg);

	// Read source image in grayscale mode
	Mat img = skinImg;

	// Apply ??? algorithm from https://stackoverflow.com/a/14874992/2501769
	Mat enhanced, float_gray, blur, num, den;
	img.convertTo(float_gray, CV_32F, 1.0 / 255.0);
	cv::GaussianBlur(float_gray, blur, Size(0, 0), 10);
	num = float_gray - blur;
	cv::GaussianBlur(num.mul(num), blur, Size(0, 0), 20);
	cv::pow(blur, 0.5, den);
	enhanced = num / den;
	cv::normalize(enhanced, enhanced, 0.0, 255.0, NORM_MINMAX, -1);
	enhanced.convertTo(enhanced, CV_8UC1);

	// Low-pass filter
	Mat gaussian;
	cv::GaussianBlur(enhanced, gaussian, Size(0, 0), 3);

	// High-pass filter on computed low-pass image
	Mat laplace;
	Laplacian(gaussian, laplace, CV_32F, 19);
	double lapmin, lapmax;
	minMaxLoc(laplace, &lapmin, &lapmax);
	double scale = 127 / max(-lapmin, lapmax);
	laplace.convertTo(laplace, CV_8U, scale, 128);

	// Thresholding using empirical value of 150 to create a vein mask
	Mat mask;
	cv::threshold(laplace, mask, THRESHOLD, 255, CV_THRESH_BINARY);

	// Clean-up the mask using open morphological operation
	morphologyEx(mask, mask, cv::MORPH_OPEN,
		getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

	// Connect the neighboring areas using close morphological operation
	Mat connected;
	morphologyEx(mask, mask, cv::MORPH_CLOSE,
		getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));

	// Blurry the mask for a smoother enhancement
	cv::GaussianBlur(mask, mask, Size(15, 15), 0);

	// Blurry a little bit the image as well to remove noise
	cv::GaussianBlur(enhanced, enhanced, Size(3, 3), 0);

	// The mask is used to amplify the veins
	Mat result(enhanced);
	ushort new_pixel;
	double coeff;
	for (int i = 0; i<mask.rows; i++) {
		for (int j = 0; j<mask.cols; j++) {
			coeff = (1.0 - (mask.at<uchar>(i, j) / 255.0))*BRIGHT + (1 - DARK);
			new_pixel = coeff * enhanced.at<uchar>(i, j);
			result.at<uchar>(i, j) = (new_pixel>255) ? 255 : new_pixel;
		}
	}


	// Show results
	imshow("frame", img);
	imshow("frame", result);
	waitKey();



	src_gray = img;

	createTrackbar(" Canny thresh:", "Source", &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);

	canny_rem = canny_dst.clone();


	cv::Mat mask2 = src_gray.clone();

	cv::dilate(mask2, mask2, cv::Mat());
	cv::dilate(mask2, mask2, cv::Mat());
	cv::erode(mask2, mask2, cv::Mat());
	cv::erode(mask2, mask2, cv::Mat());

	cv::erode(mask2, mask2, cv::Mat());
	cv::erode(mask2, mask2, cv::Mat());
	cv::dilate(mask2, mask2, cv::Mat());
	cv::dilate(mask2, mask2, cv::Mat());

	cv::Mat median;
	cv::medianBlur(img, median, 7);

	cv::Mat resizedIn;
	cv::Mat resizedMask;
	cv::Mat resizedMedian;
	cv::resize(mask2, resizedMask, cv::Size(), 1, 1);
	cv::resize(median, resizedMedian, cv::Size(), 1, 1);
	cv::resize(img, resizedIn, cv::Size(), 1, 1);

	cv::imshow("input", resizedIn);
	cv::imshow("mask", resizedMask);
	cv::imshow("median", resizedMedian);

	cv::waitKey(0);



	src_gray = median;

	createTrackbar(" Canny thresh:", "Source", &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);



	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;

	Mat thr(src.rows, src.cols, CV_8UC1);
	Mat dst(src.rows, src.cols, CV_8UC1, Scalar::all(0));
	cvtColor(src, thr, CV_BGR2GRAY); //Convert to gray
	threshold(thr, thr, 25, 255, THRESH_BINARY); //Threshold the gray

	vector<vector<Point>> contours; // Vector for storing contour
	vector<Vec4i> hierarchy;

	findContours(thr, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image

	for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a > largest_area) {
			largest_area = a;
			largest_contour_index = i;                //Store the index of largest contour
													  //bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}

	}

	Scalar color(255, 255, 255);
	drawContours(dst, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.
																						//rectangle(src, bounding_rect, Scalar(0, 255, 0), 1, 8, 0);
																						//imshow("src", src);
																						//   imshow("largest Contour", dst);

	Histogram1D h; // 히스토그램 객체
	cv::MatND histo = h.getHistogram(median); // 히스토그램 계산

	float sum = 0;
	for (int i = 0; i < 256; i++) { // 각 빈도 조회
		sum += histo.at<float>(i);
	}
	sum -= histo.at<float>(0);
	cout << "전체 다리: " << sum << endl;

	entire = sum;
	hst();


	// 영상을 두 그룹으로 나누는 부분을 경계값으로 처리해 확인
	cv::Mat thresholded; // 경계값으로 이진 영상 생성
	cv::threshold(src, thresholded, 200, 255, cv::THRESH_BINARY);
	// 영상을 경계화 하기 위해 히스토그램의 
	// 높은 봉우리(명암값 60) 방향으로 증가하기 직전인 최소값으로 정함.

	//   cv::namedWindow("Binary Image"); // 경계화된 영상 띄워 보기
	//   cv::imshow("Binary Image", thresholded); // 배경과 전경이 분할됨

	// 그레이스케일 이미지로 변환
	cvtColor(src, img_gray, COLOR_BGR2GRAY);

	//윈도우 생성  
	//   namedWindow("original image", WINDOW_AUTOSIZE);
	namedWindow("click image", WINDOW_AUTOSIZE);


	//윈도우에 출력  
	//   imshow("original image", inputImg);
	imshow("click image", inputImg);

}

int main(int argc, char** argv)
{
	thresh = 90;//170
	check_back = 0;
	//윈도우에 콜백함수를 등록
	setMouseCallback("click image", CallBackFunc, NULL);

	test();

	//윈도우에 콜백함수를 등록
	setMouseCallback("click image", CallBackFunc, NULL);

	waitKey(0);
	return 0;
}