#ifndef __SHOW_FRAME__
#define __SHOW_FRAME__
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include "../Headers/GlobalMainHeaders.h"
#include "../Models/ObjectRect.h"
#include "../Models/ResultSegment.hpp"

class ShowFrame
{
public:
	template<typename T>
	static void ToMat(T* frame, const int width, const int height, cv::Mat& img, int type);

	template<typename T>
	static void ToMat(T* frame, const int width, const int height, cv::Mat& img);

	static void Show(std::string titleName, unsigned char* frame, const int width, const int height);

	static void DrawRectangles(unsigned char* frame, ObjectRect* allRects, int width, int height);

	static inline void DrawRectangles(unsigned short* frame, ResultSegment* allRects, int width, int height, int delay = 0);

	template<typename T>
	static void ToTxt(T* frame, std::string fileName, const int width,  const int height);
};

template<typename T>
void ShowFrame::ToMat(T* frame, const int width, const int height, cv::Mat& img, int type)
{
	auto tempFrame = new unsigned char[width * height];
	memset(tempFrame, 0, sizeof(unsigned char) * width * height);
	for (auto i = 0; i < width * height; i++)
	{
		short pixelValue = frame[i];
		pixelValue >>= 0;
		char ucPixelValue = static_cast<unsigned char>(pixelValue & 0xFF);
		tempFrame[i] = static_cast<unsigned char>(ucPixelValue);
	}

	img = cv::Mat(height, width, type);
	for (auto r = 0; r < height; ++r)
	{
		auto ptr = img.ptr<uchar>(r);
		for (auto c = 0; c < width; ++c)
		{
			ptr[c] = static_cast<uchar>(tempFrame[r * width + c]);
		}
	}
	delete[] tempFrame;
}

template <typename T>
void ShowFrame::ToMat(T* frame, const int width, const int height, cv::Mat& img)
{
	for (auto r = 0; r < height; ++r)
	{
		auto ptr = img.ptr<uchar>(r);
		for (auto c = 0; c < width; ++c)
		{
			ptr[c] = static_cast<uchar>(frame[r * width + c]);
		}
	}
}

template<typename T>
void ShowFrame::ToTxt(T* frame, std::string fileName, const int width, const int height)
{
	std::ofstream fout(fileName);

	if (fout.is_open())
	{
		logPrinter.PrintLogs("Write frame data to text file...", LogLevel::Info);
		for (auto i = 0; i < height; ++i)
		{
			for (auto j = 0; j < width; ++j)
			{
				fout << std::setw(6) << static_cast<int>(frame[i * width + j]) << " ";
			}
			fout << std::endl;
		}
		logPrinter.PrintLogs("Write frame data to text file is done!", Info);
	}
	else
	{
		logPrinter.PrintLogs("Since cannot open this file, Wtite pixels data to text file faild", Error);
	}
}

inline void ShowFrame::Show(std::string titleName, unsigned char* frame, const int width, const int height)
{
	cv::Mat img;
	ToMat<unsigned char>(frame, width, height, img, CV_8UC1);

	imshow(titleName, img);
	cv::waitKey(0);
}

inline void ShowFrame::DrawRectangles(unsigned char* frame, ObjectRect* allRects, int width, int height)
{
	cv::Mat img;
	ToMat<unsigned char>(frame, width, height, img, CV_8UC1);

	for (auto i = 0; i < width * height; ++i)
	{
		if (allRects[i].width < 2 || allRects[i].height < 2 || allRects[i].width > 20 || allRects[i].height > 20)
		{
			continue;
		}
		rectangle(img, cv::Point(allRects[i].lt.x, allRects[i].lt.y), cv::Point(allRects[i].rb.x + 1, allRects[i].rb.y + 1), cv::Scalar(255, 0, 0));
	}

	cv::imshow("all candiates draw", img);
	cv::waitKey(1);
}

inline void ShowFrame::DrawRectangles(unsigned short* frame, ResultSegment* allRects, int width, int height, int delay)
{
	cv::Mat img;
	ToMat<unsigned short>(frame, width, height, img, CV_8UC1);

	for (auto i = 0; i < allRects->targetCount; ++i)
	{
		rectangle(img, cv::Point(allRects->targets[i].topLeftX, allRects->targets[i].topleftY), cv::Point(allRects->targets[i].bottomRightX + 1, allRects->targets[i].bottomRightY + 1), cv::Scalar(255, 255, 0));
	}

	imshow("after draw", img);
	cv::waitKey(delay);
}
#endif
