#ifndef __SHOW_FRAME__
#define __SHOW_FRAME__
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include "../Headers/GlobalMainHeaders.h"

class ShowFrame
{
public:
	template<typename T>
	static void ToMat(T* frame, const int width, const int height, cv::Mat& img, int type);

	static void Show(std::string titleName, unsigned char* frame, const int width, const int height);

	static void ToTxt(unsigned char* frame, std::string fileName, const int width,  const int height);
};

template<typename T>
void ShowFrame::ToMat(T* frame, const int width, const int height, cv::Mat& img, int type)
{
	img = cv::Mat(height, width, type);
	for (auto r = 0; r < height; ++r)
	{
		auto ptr = img.ptr<uchar>(r);
		for (auto c = 0; c < width; ++c)
		{
			ptr[c] = static_cast<uchar>(frame[r * width + c]);
		}
	}
}

inline void ShowFrame::ToTxt(unsigned char* frame, std::string fileName, const int width, const int height)
{
	std::ofstream fout(fileName);

	if (fout.is_open())
	{
		for (auto i = 0; i < height; ++i)
		{
			for (auto j = 0; j < width; ++j)
			{
				fout << static_cast<int>(frame[i * width + j]) << " ";
			}
			fout << std::endl;
		}
	}
	else
	{
		logPrinter.PrintLogs("Since cannot open this file, Wtite pixels data to text file faild", Error);
	}
}

inline void ShowFrame::Show(std::string titleName, unsigned char* frame, const int width, const int height)
{
	cv::Mat img;
	ToMat(frame, width, height, img, CV_8UC1);

	imshow(titleName, img);
	cv::waitKey(0);
}
#endif
