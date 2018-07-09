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
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		cout << "# of contour points: " << contours[i].size() << std::endl;

		for (unsigned int j = 0; j<contours[i].size(); j++)
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


		for (int h = 0; h<histSize[0]; h++) { // �� �󵵿� ���� �������� �׸��� 
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
	Mat hlsImg;
	Mat skinImg;
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

	for (int i = 0; i< contours.size(); i++) // iterate through each contour. 
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a>largest_area) {
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

	waitKey(0);
	return 0;
}