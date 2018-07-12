#pragma once
//====================================================watershedSegment.h

class WatershedSegmenter {

private:

	cv::Mat markers;

public:

	void setMarkers(const cv::Mat& markerImage) {

		markerImage.convertTo(markers, CV_32S); //32��Ʈ��Ŀ�����ڷ�����ȯ

	}

	cv::Mat process(const cv::Mat& image) {

		cv::watershed(image, markers);

		//���Ұ����markers������

		return markers;

	}

	cv::Mat getSegmentation() {

		cv::Mat tmp;

		markers.convertTo(tmp, CV_8U); return tmp;

	}

	cv::Mat getWatersheds() {

		cv::Mat tmp;

		markers.convertTo(tmp, CV_8U, 255, 255); return tmp;

	}

};

