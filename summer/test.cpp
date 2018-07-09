//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <opencv/highgui.h>
//#include <opencv/cv.h>
//#include <stdio.h>
//#include <stdlib.h>
//
//using namespace cv;
//using namespace std;
//
//
//Mat src; Mat src_gray;
//int thresh = 40;
//int max_thresh = 255;
//RNG rng(12345);
//int ans;
//
//
//int main(int argc, char** argv)
//{
//	Mat rgbImg;
//	Mat veinImg;
//	
//
//	veinImg = imread("city.jpg");
//	CV_Assert(veinImg.depth() == CV_8U);
//	//resultImg.create(inputImg.size(), inputImg.type());
//
//
//	//veinImg = skinImg.clone();
//	cvtColor(veinImg, rgbImg, CV_BGR2HLS);
//	vector<Mat> rgb_images(3);
//	split(rgbImg, rgb_images);
//
//	for (int row = 0; row < rgbImg.rows; row++)
//	{
//		for (int col = 0; col < rgbImg.cols; col++)
//		{
//			uchar b = rgbImg.at<Vec3b>(row, col)[0];
//			uchar g = rgbImg.at<Vec3b>(row, col)[1];
//			uchar r = rgbImg.at<Vec3b>(row, col)[2];
//
//			bool vein_pixel =(r >= 66) && (g < 97) && (b < 107);
//
//			if (vein_pixel == false)
//			{
//				veinImg.at<Vec3b>(row, col)[0] = 0;
//				veinImg.at<Vec3b>(row, col)[1] = 0;
//				veinImg.at<Vec3b>(row, col)[2] = 0;
//			}
//		}
//	}
//
//	namedWindow("VeinDetected", CV_WINDOW_AUTOSIZE);
//	moveWindow("VeinDetected", 120, 120);
//	imshow("VeinDetected", veinImg);
//
//
//	waitKey(0);
//	return 0;
//}
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//
//#define THRESHOLD 100
//#define BRIGHT 0.7
//#define DARK 0.2
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//
//	// Read source image in grayscale mode
//	Mat img = imread("city.png");
//
//	// Apply ??? algorithm from https://stackoverflow.com/a/14874992/2501769
//	Mat enhanced, float_gray, blur, num, den;
//	img.convertTo(float_gray, CV_32F, 1.0 / 255.0);
//	cv::GaussianBlur(float_gray, blur, Size(0, 0), 10);
//	num = float_gray - blur;
//	cv::GaussianBlur(num.mul(num), blur, Size(0, 0), 20);
//	cv::pow(blur, 0.5, den);
//	enhanced = num / den;
//	cv::normalize(enhanced, enhanced, 0.0, 255.0, NORM_MINMAX, -1);
//	enhanced.convertTo(enhanced, CV_8UC1);
//
//	// Low-pass filter
//	Mat gaussian;
//	cv::GaussianBlur(enhanced, gaussian, Size(0, 0), 3);
//
//	// High-pass filter on computed low-pass image
//	Mat laplace;
//	Laplacian(gaussian, laplace, CV_32F, 19);
//	double lapmin, lapmax;
//	minMaxLoc(laplace, &lapmin, &lapmax);
//	double scale = 127 / max(-lapmin, lapmax);
//	laplace.convertTo(laplace, CV_8U, scale, 128);
//
//	// Thresholding using empirical value of 150 to create a vein mask
//	Mat mask;
//	cv::threshold(laplace, mask, THRESHOLD, 255, CV_THRESH_BINARY);
//
//	// Clean-up the mask using open morphological operation
//	morphologyEx(mask, mask, cv::MORPH_OPEN,
//		getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
//
//	// Connect the neighboring areas using close morphological operation
//	Mat connected;
//	morphologyEx(mask, mask, cv::MORPH_CLOSE,
//		getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
//
//	// Blurry the mask for a smoother enhancement
//	cv::GaussianBlur(mask, mask, Size(15, 15), 0);
//
//	// Blurry a little bit the image as well to remove noise
//	cv::GaussianBlur(enhanced, enhanced, Size(3, 3), 0);
//
//	// The mask is used to amplify the veins
//	Mat result(enhanced);
//	ushort new_pixel;
//	double coeff;
//	for (int i = 0; i<mask.rows; i++) {
//		for (int j = 0; j<mask.cols; j++) {
//			coeff = (1.0 - (mask.at<uchar>(i, j) / 255.0))*BRIGHT + (1 - DARK);
//			new_pixel = coeff * enhanced.at<uchar>(i, j);
//			result.at<uchar>(i, j) = (new_pixel>255) ? 255 : new_pixel;
//		}
//	}
//
//	// Show results
//	imshow("frame", img);
//	waitKey();
//
//	imshow("frame", result);
//	waitKey();
//
//	return 0;
//}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


Mat src; Mat src_gray;
int thresh = 40;
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

	Canny(src_gray, canny_output, thresh, thresh * 2, 3);

	namedWindow(src_name, WINDOW_AUTOSIZE);
	namedWindow(canny_name, WINDOW_AUTOSIZE);

	//This is for source image.
	imshow(src_name, src);

	Mat canny_dst;
	//This is for canny edge detection.
	Canny(src_gray, canny_dst, 50, 200, 3, false);
	imshow(canny_name, canny_dst);

	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw contours
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		cout << "# of contour points: " << contours[i].size() << std::endl;

		for (unsigned int j = 0; j < contours[i].size(); j++)
		{
			ans += contourArea(contours[i]);
			//cout << "Point(x,y)=" << contours[i][j] << std::endl;
		}

		cout << " Area: " << i << " " << contourArea(contours[i]) << "\n\n";
	}
	cout << "part of Area: " << ans << "\n\n";
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "���� ���콺 ��ư Ŭ��.. ��ǥ = (" << x << ", " << y << ")" << endl;
	}
}

class Histogram1D {
private:
	int histSize[1]; // �󵵼�
	float hranges[2]; // �ּ�/�ִ� ȭ�Ұ�
	const float* ranges[1];
	int channels[1]; // ���⼭ 1ä�θ� ���
public:
	Histogram1D() { // 1���� ������׷��� ���� ���� �غ�
		histSize[0] = 256;
		hranges[0] = 0.0;
		hranges[1] = 255.0;
		ranges[0] = hranges;
		channels[0] = 0; // �⺻������ ä���� 0���� ����
	}

	// ������ ��� ������ �׷��̷��� ������ ������׷��� ����� ���� ���� �޼ҵ带 ����� ����
	cv::MatND getHistogram(const cv::Mat &image) {
		// 1����(1D) ������׷� ���.
		cv::MatND hist;
		cv::calcHist(&image, // ������׷� ��� 
			1,   // ���� ������ ������׷���
			channels, // ��� ä��               
			cv::Mat(), // ����ũ ������� ����     
			hist,  // ��� ������׷�         
			1,   // 1����(1D) ������׷�           
			histSize, // �󵵼�                  
			ranges  // ȭ�Ұ� ����             
		);
		return hist;
	}

	// ���� ���������� �ǹ̸� �ľ��ϱ� �����Ƿ� �� �׷����� ���
	// �׷����� �����ϴ� �޼ҵ�
	cv::Mat getHistogramImage(const cv::Mat &image) {
		// 1����(1D) ������׷��� ����ϰ�, �������� ��ȯ

		cv::MatND hist = getHistogram(image); // ���� ������׷� ���

		double maxVal = 0; // �ִ� �󵵼� ��������
		double minVal = 0; // �ּ� �󵵼� ��������
		cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

		cv::Mat histImg(histSize[0], histSize[0], CV_8U, cv::Scalar(255));
		// ������׷��� ����ϱ� ���� ����

		int hpt = static_cast<int>(0.9*histSize[0]);
		// nbins�� 90%�� �ִ������� ����


		for (int h = 0; h < histSize[0]; h++) { // �� �󵵿� ���� �������� �׸��� 
			float binVal = hist.at<float>(h);
			int intensity = static_cast<int>(binVal*hpt / maxVal);
			cv::line(histImg, cv::Point(h, histSize[0]), cv::Point(h, histSize[0] - intensity), cv::Scalar::all(0));
			// �� �� ���� �Ÿ��� �׸��� �Լ�
		}
		return histImg;
	}
};

int main(int argc, char** argv)
{
	Mat inputImg, resultImg;
	Mat hlsImg, rgbImg;
	Mat skinImg, veinImg;
	Mat img_gray;

	inputImg = imread("city5.jpg");
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


	/// Convert image to gray and blur it
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	/// Create Window

	src = skinImg;
	const char* source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, src);

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
	imshow("largest Contour", dst);

	//������׷�
	if (!src.data)
		return 0;

	cv::namedWindow("Image");
	cv::imshow("Image", src);

	Histogram1D h; // ������׷� ��ü
	cv::MatND histo = h.getHistogram(src); // ������׷� ���

	int sum = 0;
	for (int i = 0; i < 256; i++) { // �� �� ��ȸ
									//   std::cout << "Value " << i << " = " << histo.at<float>(i) << std::endl;
		sum += histo.at<float>(i);
	}

	cout << sum - histo.at<float>(0) << endl;

	cv::namedWindow("Histogram");
	cv::imshow("Histogram", h.getHistogramImage(src));
	// ������׷��� �������� ����
	// ����� �߽����� ������ ������, �������� �����
	// ��� ���츮 �κ��� �߰� ��ϵ� ��
	// ������ ������ ����, �������� ���

	// ������ �� �׷����� ������ �κ��� ��谪���� ó���� Ȯ��
	cv::Mat thresholded; // ��谪���� ���� ���� ����
	cv::threshold(src, thresholded, 60, 255, cv::THRESH_BINARY);
	// ������ ���ȭ �ϱ� ���� ������׷��� 
	// ���� ���츮(��ϰ� 60) �������� �����ϱ� ������ �ּҰ����� ����.
	cv::namedWindow("Binary Image"); // ���ȭ�� ���� ��� ����
	cv::imshow("Binary Image", thresholded); // ���� ������ ���ҵ�

											 // �׷��̽����� �̹����� ��ȯ
	cvtColor(src, img_gray, COLOR_BGR2GRAY);

	//������ ����  
	namedWindow("original image", WINDOW_AUTOSIZE);
	namedWindow("gray image", WINDOW_AUTOSIZE);


	//�����쿡 ���  
	imshow("original image", src);
	imshow("gray image", img_gray);

	//�����쿡 �ݹ��Լ��� ���
	setMouseCallback("gray image", CallBackFunc, NULL);

	// read a pixel from image directly // pixel���־����� �� pixel�� �� ����
	//int nBlue, nGreen, nRed;
	//nBlue = 0; nGreen = 0; nRed = 0;

	//nBlue = inputImg.at<cv::Vec3b>(167, 342)[0];
	//nGreen = inputImg.at<cv::Vec3b>(167, 342)[1];
	//nRed = inputImg.at<cv::Vec3b>(167, 342)[2];

	//cout << "Red : " << nRed << " , Green : " << nGreen << " , Blue : " << nBlue << endl;

	//veinImg = skinImg.clone();

	//cvtColor(veinImg, rgbImg, CV_BGR2HLS);
	//vector<Mat> rgb_images(3);
	//split(rgbImg, rgb_images);

	//for (int row = 0; row < rgbImg.rows; row++)
	//{
	//	for (int col = 0; col < rgbImg.cols; col++)
	//	{
	//		uchar b = rgbImg.at<Vec3b>(row, col)[0];
	//		uchar g = rgbImg.at<Vec3b>(row, col)[1];
	//		uchar r = rgbImg.at<Vec3b>(row, col)[2];

	//		bool vein_pixel = (r <= 50) && (g < 30) && (b < 110);

	//		if (vein_pixel == false)
	//		{
	//			veinImg.at<Vec3b>(row, col)[0] = 0;
	//			veinImg.at<Vec3b>(row, col)[1] = 0;
	//			veinImg.at<Vec3b>(row, col)[2] = 0;
	//		}
	//	}
	//}

	//namedWindow("Original", CV_WINDOW_AUTOSIZE);
	//namedWindow("VeinDetected", CV_WINDOW_AUTOSIZE);


	//moveWindow("Original", 100, 100);
	//moveWindow("VeinDetected", 120, 120);


	//imshow("Original", src);
	//imshow("VeinDetected", veinImg);


	waitKey(0);
	return 0;
}
