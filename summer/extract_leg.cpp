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
Mat area1, area2;

int check;
Mat src; Mat src_gray;
int thresh = 90;//170
int max_thresh = 255;
RNG rng(12345);
int ans;

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


	Canny(src_gray, canny_output, thresh, thresh * 2, 3);

	Canny(src_gray, canny_dst, 50, 200, 3, false);

	/// Find contours 
	findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/*canny_dst = canny_dst.clone();

	if (check == 1)
	imshow(canny_name, canny_dst);
	else
	imshow(canny_name2, canny_dst);
	*/

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

		//cout << " Area: " << i << " " << contourArea(contours[i]) << "\n\n";
	}


	/// Show in a window 

	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);

	//   findContours(canny_output, canny_dst, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	canny_dst = drawing.clone();

	if (check == 1) {
		imshow(canny_name, canny_dst);
		area1 = canny_dst;
	}
	else{
		imshow(canny_name2, canny_dst);
		area2 = canny_dst;
	}

	check++;
}




void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "왼쪽 마우스 버튼 클릭.. 좌표 = (" << x << ", " << y << ")" << endl;
	}
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

int main(int argc, char** argv)
{
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

		//if (contours[i].size() < 10000 && contours[i].size() > 0)
		//{
		//   double size = cv::contourArea(contours[i]);
		//   if (size>largest_area)
		//   {
		//      largest_area = size;
		//      largest_contour_index = i; 
		//   }
		//}
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

																						//히스토그램
	if (!skinImg.data)
		return 0;

	//   cv::namedWindow("Image");
	//   cv::imshow("Image", skinImg);

	Histogram1D h; // 히스토그램 객체
	cv::MatND histo = h.getHistogram(median); // 히스토그램 계산

	float sum = 0;
	for (int i = 0; i < 256; i++) { // 각 빈도 조회
									//   std::cout << "Value " << i << " = " << histo.at<float>(i) << std::endl;
		sum += histo.at<float>(i);
		//cout << i << " is " << histo.at<float>(i)<<endl;
	}
	sum -= histo.at<float>(0);
	cout << "전체 다리: " << sum<< endl;

	Histogram1D h1; // 히스토그램 객체
	cv::MatND histo1 = h1.getHistogram(area1); // 히스토그램 계산

	float sum1=0;
	for (int i = 0; i < 256; i++) { // 각 빈도 조회
									//   std::cout << "Value " << i << " = " << histo.at<float>(i) << std::endl;
		sum1 += histo1.at<float>(i);
		//cout << i << " is " << histo1.at<float>(i)<<endl;
	}
//	sum1 -= histo1.at<float>(0);
	Histogram1D h2; // 히스토그램 객체
	cv::MatND histo2 = h2.getHistogram(area2); // 히스토그램 계산

	float sum2=0;
	for (int i = 0; i < 256; i++) { // 각 빈도 조회
									//   std::cout << "Value " << i << " = " << histo.at<float>(i) << std::endl;
		sum2 += histo2.at<float>(i);

		//cout << i << " is " << histo1.at<float>(i) << endl;
	}
//	sum2 -= histo2.at<float>(0);

	//cout << "다리영역 sum1 << "  " << sum2<<endl;
	float sum3 = sum2 - sum1;
	cout << "하지정맥 영역: " <<sum3<<endl;
	cout << "하지정맥이 차지하는 퍼센트" << ( sum3 / sum )* 100 << endl;



	//   cv::namedWindow("Histogram");
	//   cv::imshow("Histogram", h.getHistogramImage(src));
	// 히스토그램을 영상으로 띄우기
	// 가운데를 중심으로 왼쪽이 검정색, 오른쪽이 흰색값
	// 가운데 봉우리 부분은 중간 명암도 값
	// 왼쪽이 영상의 전경, 오른쪽이 배경

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

	//윈도우에 콜백함수를 등록
	setMouseCallback("gray image", CallBackFunc, NULL);


	waitKey(0);
	return 0;
}