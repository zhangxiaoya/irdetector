#ifndef __SHOW_FRAME__
#define __SHOW_FRAME__
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class ShowFrame
{
public:
	static void Show(std::string titleName, unsigned char* frame, const int width, const int height);
};

inline void ShowFrame::Show(std::string titleName, unsigned char* frame, const int width, const int height)
{
	cv::Mat img(height, width, CV_8UC1);
	for (auto r = 0; r < height; ++r)
	{
		auto ptr = img.ptr<uchar>(r);
		for (auto c = 0; c < width; ++c)
		{
			ptr[c] = static_cast<uchar>(frame[r * width + c]);
		}
	}

	imshow(titleName, img);
	cv::waitKey(0);
}
#endif
