#pragma once
#include "../DataReaderFromeFiles/BinaryFileReader.hpp"
#include "../Models/LogLevel.hpp"
#include "../LogPrinter/LogPrinter.hpp"
#include "../Checkers/CheckPerf.h"
#include "../Detector/Detector.hpp"
#include "../Models/CandidateTargets.hpp"


bool CompareCandidates(CandidateTarget& a, CandidateTarget& b)
{
	return  a.score > b.score;
}

class SearchValidation
{
public:
	explicit SearchValidation(const int width,
	                          const int height,
	                          const int pixelSize,
	                          const int dilationRadius,
	                          const int discretizationScale,
	                          BinaryFileReader* file_reader = nullptr)
		: fileReader(file_reader),
		  detector(nullptr),
		  Width(width),
		  Height(height),
		  PixelSize(pixelSize),
		  DilationRadius(dilationRadius),
		  DiscretizationScale(discretizationScale),
		  candidateTargetCount(0)
	{
	}

	~SearchValidation()
	{
		delete fileReader;
	}

	void InitDataReader(std::string validationFileName);

	bool CheckGround(int frame_index) const;

	void ShowFirstFive(unsigned short** dataPoint);

	void VailidationAll();

private:
	bool CheckFileReader() const;

	double CalculateSCR(unsigned short* frame, const TargetPosition& target);

	double CalculateMaxValue(unsigned short* frame, const TargetPosition& target) const;

	void CalculateScoreForDetectedTargetsAndPushToCandidateQueue(unsigned short* frame, const DetectResultSegment& result, int frameIndex);

	BinaryFileReader* fileReader;
	Detector* detector;

	int Width;
	int Height;
	int PixelSize;
	int DilationRadius;
	int DiscretizationScale;

	LogPrinter logPrinter;

	CandidateTarget AllCandidateTargets[5 * 171];
	int candidateTargetCount;
	int frameIndexw;
};

inline void SearchValidation::InitDataReader(const std::string validationFileName)
{
	if (fileReader != nullptr)
	{
		delete fileReader;
		fileReader = nullptr;
	}
	fileReader = new BinaryFileReader(Width, Height, PixelSize, validationFileName);
	fileReader->ReadBinaryFileToHostMemory();
}

inline bool SearchValidation::CheckGround(int frame_index) const
{
	int beg = 364;
	int end = 445;
	while (beg < 3467)
	{
		if(frame_index >= beg && frame_index < end)
			return true;
		beg += 171;
		end += 171;
	}
	return false;
}

inline void SearchValidation::ShowFirstFive(unsigned short** dataPoint)
{
	std::sort(this->AllCandidateTargets, AllCandidateTargets + candidateTargetCount, CompareCandidates);
	int flag[5] = { -1 };
	cv::Mat imgs[5];

	int targetCount = 0;
	int frameCount = 0;
	while(frameCount < 5)
	{
		bool existFlag = false;
		for(int i = 0;i < frameCount;++i)
		{
			if (flag[i] == AllCandidateTargets[targetCount].frameIndex)
			{
				existFlag = true;
				break;
			}
		}
		if(existFlag == true)
		{
			targetCount++;
			continue;
		}
		
		flag[frameCount] = AllCandidateTargets[targetCount].frameIndex;
		frameCount++;
		targetCount++;
	}
	
	for(int i = 0; i< frameCount; ++i)
	{
		bool initImage = false;
		for(int j = 0; j< targetCount;++j)
		{
			if(AllCandidateTargets[j].frameIndex == flag[i] && initImage == false)
			{
				ShowFrame::ToMat<unsigned short>(dataPoint[AllCandidateTargets[j].frameIndex], Width, Height, imgs[i], CV_8UC3);
				initImage = true;
			}
			if(AllCandidateTargets[j].frameIndex == flag[i])
				rectangle(imgs[i], cv::Point(AllCandidateTargets[j].left -5 , AllCandidateTargets[j].top- 5), cv::Point(AllCandidateTargets[j].right + 5, AllCandidateTargets[j].bottom + 5), cv::Scalar(255, 255, 0));
		}
		std::cout << "index is " << std::setw(4) << flag[i] << std::endl;
		imshow("after draw", imgs[i]);
		cv::waitKey(0);
	}
	const int fileNameBufferSize = 200;
	char outputFrameName[fileNameBufferSize];
	for(int i = 0; i< 5; i++)
	{
		sprintf_s(outputFrameName, fileNameBufferSize, "D:\\Cabins\\Projects\\Project1\\Frame_%04d.png", frameIndexw);
		imwrite(outputFrameName, imgs[i]);
		frameIndexw++;
	}
	
}

inline bool SearchValidation::CheckFileReader() const
{
	if (fileReader == nullptr)
	{
		logPrinter.PrintLogs("File Reader is Not Ready!", Error);
		system("Pause");
		return true;
	}
	return false;
}

inline double SearchValidation::CalculateSCR(unsigned short* frame, const TargetPosition& target)
{
	int width = target.bottomRightX - target.topLeftX;
	int height = target.bottomRightY - target.topLeftY;

	int widthPadding = width;
	int heightPadding = height;

	double avgTarget = 0.0;
	double avgSurrouding = 0.0;

	// target average gray value
	double sum = 0.0;
	for (int r = target.topLeftY; r < target.bottomRightY; ++r)
	{
		double sumRow = 0.0;
		for (int c = target.topLeftX; c < target.bottomRightX; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += (sumRow / width);
	}
	avgTarget = sum / height;

	// target surrounding average gray value
	sum = 0.0;
	int surroundingTop = target.topLeftY - heightPadding;
	surroundingTop = surroundingTop < 0 ? 0 : surroundingTop;
	int surroundingLeft = target.topLeftX - widthPadding;
	surroundingLeft = surroundingLeft < 0 ? 0 : surroundingLeft;
	int surroundingRight = target.bottomRightX + widthPadding;
	surroundingRight = surroundingRight > Width ? Width : surroundingRight;
	int surroundingBottom = target.bottomRightY + heightPadding;
	surroundingBottom = surroundingBottom > Height ? Height : surroundingBottom;
	for (int r = surroundingTop; r < target.topLeftY; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (int r = target.bottomRightY; r < surroundingBottom; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (int r = target.topLeftY; r < target.bottomRightY; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < target.topLeftX; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		for (int c = target.bottomRightX; c < surroundingRight; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += sumRow / ((target.topLeftX - surroundingLeft) + (surroundingRight - target.bottomRightX));
	}
	avgSurrouding = sum / (surroundingBottom - surroundingTop);

	// calculate standard deviation
	sum = 0.0;
	for (int r = surroundingTop; r < target.topLeftY; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += (static_cast<double>(frame[r * Width + c]) - avgSurrouding) * (static_cast<double>(frame[r * Width + c]) - avgSurrouding);
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (int r = target.bottomRightY; r < surroundingBottom; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += (static_cast<double>(frame[r * Width + c]) - avgSurrouding) * (static_cast<double>(frame[r * Width + c]) - avgSurrouding);
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (int r = target.topLeftY; r < target.bottomRightY; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < target.topLeftX; ++c)
		{
			sumRow += (static_cast<double>(frame[r * Width + c]) - avgSurrouding) * (static_cast<double>(frame[r * Width + c]) - avgSurrouding);
		}
		for (int c = target.bottomRightX; c < surroundingRight; ++c)
		{
			sumRow += (static_cast<double>(frame[r * Width + c]) - avgSurrouding) * (static_cast<double>(frame[r * Width + c]) - avgSurrouding);
		}
		sum += sumRow / ((target.topLeftX - surroundingLeft) + (surroundingRight - target.bottomRightX));
	}

	double stdDivation = sqrt(sum / (surroundingBottom - surroundingTop));

	return abs(avgTarget - avgSurrouding) / stdDivation;
}

inline double SearchValidation::CalculateMaxValue(unsigned short* frame, const TargetPosition& target) const
{
	double max = 0.0;
	for(auto r = target.topLeftY; r < target.bottomRightY; ++r)
	{
		for(auto c = target.topLeftX; c < target.bottomRightX; ++c)
		{
			if (max < frame[r * Width + c])
				max = static_cast<double>(frame[r * Width + c]);
		}
	}
	return max;
}

inline void SearchValidation::CalculateScoreForDetectedTargetsAndPushToCandidateQueue(unsigned short* frame, const DetectResultSegment& result, int frameIndex)
{
	for(auto i = 0; i< result.targetCount; i++)
	{
//		double score = CalculateSCR(frame, result.targets[i]);
		double score = CalculateMaxValue(frame, result.targets[i]);

		AllCandidateTargets[candidateTargetCount].left = result.targets[i].topLeftX;
		AllCandidateTargets[candidateTargetCount].right = result.targets[i].bottomRightX;
		AllCandidateTargets[candidateTargetCount].top = result.targets[i].topLeftY;
		AllCandidateTargets[candidateTargetCount].bottom = result.targets[i].bottomRightY;
		AllCandidateTargets[candidateTargetCount].frameIndex = frameIndex;
		AllCandidateTargets[candidateTargetCount].score = score;
		candidateTargetCount++;
	}
}

inline void SearchValidation::VailidationAll()
{
	if (CheckFileReader()) return;
	
	frameIndexw = 0;
	this->detector = new Detector(Width, Height, DilationRadius, DiscretizationScale);
	detector->InitSpace();
	detector->SetRemoveFalseAlarmParameters(true, false, false, false, true, true);

	const auto frameCount = fileReader->GetFrameCount();
	auto dataPoint = fileReader->GetDataPoint();

	logPrinter.PrintLogs("Test the detect result during 360 search... ", Info);
	char iterationText[200];

	DetectResultSegment result;
	detector->SetRemoveFalseAlarmParameters(true, false, false, false, true, true);


	for (unsigned i = 0; i < frameCount; ++i)
	{
		if(i < 300)
			continue;

		if(i != 300 && (i - 300) % 171 == 0 )
		{
			ShowFirstFive(dataPoint);
			candidateTargetCount = 0;
		}

		sprintf_s(iterationText, 200, "Checking for frame %04d ...", i);
		logPrinter.PrintLogs(iterationText, Info);

		if (CheckGround(i) == true)
		{
			continue;
		}
		CheckPerf(detector->DetectTargets(dataPoint[i], &result, nullptr, nullptr), "whole");

		CalculateScoreForDetectedTargetsAndPushToCandidateQueue(dataPoint[i], result, i);

		ShowFrame::DrawRectangles(dataPoint[i], &result, Width, Height, 1);
	}
}
