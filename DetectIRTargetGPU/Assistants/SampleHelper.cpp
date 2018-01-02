#include "SampleHelper.h"

void SampleHelper::RetryNegativeSampleFileOut()
{
	negativeFout.seekp(negativeFout.cur - 1280);
}

void SampleHelper::RetryPositiveSampleFileOut()
{
	positiveFout.seekp(positiveFout.cur - 1280);
}


void SampleHelper::InitPositiveSampleFileStream()
{
	if (positiveFout.is_open() == false)
	{
		positiveFout.open("PostiveSamples.data", std::ios::out | std::ios::app);
	}
}

void SampleHelper::SavePositiveSample(cv::Mat& sampleTarget)
{
	InitPositiveSampleFileStream();
	for(auto r = 0; r < sampleTarget.rows; ++ r)
	{
		for(auto c = 0; c < sampleTarget.cols; ++c)
		{
			positiveFout << static_cast<int>(sampleTarget.at<unsigned short>(r, c)) << " ";
		}
	}
	positiveFout << std::endl;
}

void SampleHelper::InitNegativeSampleFileStream()
{
	if (negativeFout.is_open() == false)
	{
		negativeFout.open("NegtivateSamples.data", std::ios::out | std::ios::app);
	}
}

void SampleHelper::SaveNegativeSample(cv::Mat& sampleTarget)
{
	InitNegativeSampleFileStream();
	for(auto r = 0; r < sampleTarget.rows;++r)
	{
		for(auto c = 0; c < sampleTarget.cols; ++ c)
		{
			negativeFout << static_cast<int>(sampleTarget.at<unsigned short>(r, c)) << " ";
		}
	}
	negativeFout << std::endl;
}

void SampleHelper::InitAllFileStream()
{
	InitPositiveSampleFileStream();
	InitNegativeSampleFileStream();
}

void SampleHelper::ReleasAllFileStream()
{
	if (positiveFout.is_open() == true)
	{
		positiveFout.close();
	}
	if (negativeFout.is_open() == true)
	{
		negativeFout.close();
	}
}
