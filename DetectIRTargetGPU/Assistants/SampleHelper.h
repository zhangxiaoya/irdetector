#ifndef __SAMPLE_HELPER_H__
#define __SAMPLE_HELPER_H__
#include <fstream>
#include <core/core.hpp>

static std::ofstream positiveFout;
static std::ofstream negativeFout;

class SampleHelper
{
public:
	static void RetryPositiveSampleFileOut();
	static void RetryNegativeSampleFileOut();

	static void InitPositiveSampleFileStream();
	static void InitNegativeSampleFileStream();

	static void SavePositiveSample(cv::Mat& sampleTarget);
	static void SaveNegativeSample(cv::Mat& sampleTarget);

	static void InitAllFileStream();
	static void ReleasAllFileStream();
};
#endif
