#pragma once
#include "../DataReaderFromeFiles/BinaryFileReader.hpp"
#include "../Models/LogLevel.hpp"
#include "../LogPrinter/LogPrinter.hpp"
#include "../Checkers/CheckCUDAReturnStatus.h"
#include "../CCL/MeshCCLOnCPU.hpp"
#include "../Checkers/CheckPerf.h"
#include "../Detector/Detector.hpp"

class PerformanceValidation
{
public:
	explicit PerformanceValidation(const int width,
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
		  DiscretizationScale(discretizationScale)
	{
	}

	~PerformanceValidation()
	{
		delete fileReader;
	}

	void InitDataReader(std::string validationFileName);

	void VailidationAll();

private:
	bool CheckFileReader() const;

	BinaryFileReader* fileReader;
	Detector* detector;

	int Width;
	int Height;
	int PixelSize;
	int DilationRadius;
	int DiscretizationScale;

	LogPrinter logPrinter;
};

inline void PerformanceValidation::InitDataReader(const std::string validationFileName)
{
	if(fileReader != nullptr)
	{
		delete fileReader;
		fileReader = nullptr;
	}
	fileReader = new BinaryFileReader(Width, Height, PixelSize, validationFileName);
	fileReader->ReadBinaryFileToHostMemory();
}

inline bool PerformanceValidation::CheckFileReader() const
{
	if(fileReader == nullptr)
	{
		logPrinter.PrintLogs("File Reader is Not Ready!", Error);
		system("Pause");
		return true;
	}
	return false;
}

inline void PerformanceValidation::VailidationAll()
{
	if (CheckFileReader()) return;

	this->detector = new Detector(Width, Height, DilationRadius, DiscretizationScale);
	detector->InitSpace();
	detector->SetRemoveFalseAlarmParameters(true, false, false, false, true, true);

	const auto frameCount = fileReader->GetFrameCount();
	auto dataPoint = fileReader->GetDataPoint();

	logPrinter.PrintLogs("Test the visual effect of detect result ... ", Info);
	char iterationText[200];

	ResultSegment result;
	detector->SetRemoveFalseAlarmParameters(true, false, false, false, true, true);

	for(unsigned i = 0;i<frameCount;++i)
	{
		sprintf_s(iterationText, 200, "Checking for frame %04d ...", i);
		logPrinter.PrintLogs(iterationText, Info);

		CheckPerf(detector->DetectTargets(dataPoint[i], &result), "whole");

		ShowFrame::DrawRectangles(dataPoint[i], &result, Width, Height, 1);
	}
}
