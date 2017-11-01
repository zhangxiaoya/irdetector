#pragma once
#include "../DataReaderFromeFiles/BinaryFileReader.hpp"
#include "../Models/LogLevel.hpp"
#include "../LogPrinter/LogPrinter.hpp"
#include "../Checkers/CheckCUDAReturnStatus.h"
#include "../CCL/MeshCCLOnCPU.hpp"
#include "../Checkers/CheckPerf.h"
#include "../Detector/Detector.hpp"
#include "../Monitor/Monitor.hpp"

class TrackingValidation
{
public:
	TrackingValidation(const int width,
	                            const int height,
	                            const int pixelSize,
	                            const int dilationRadius,
	                            const int discretizationScale,
	                            BinaryFileReader* file_reader = NULL)
		: fileReader(file_reader),
		  monitor(NULL),
		  Width(width),
		  Height(height),
		  PixelSize(pixelSize),
		  DilationRadius(dilationRadius),
		  DiscretizationScale(discretizationScale)
	{
	}

	~TrackingValidation()
	{
		delete fileReader;
	}

	void InitDataReader(std::string validationFileName);

	void VailidationAll();

private:
	bool CheckFileReader() const;

	BinaryFileReader* fileReader;
	Monitor* monitor;

	int Width;
	int Height;
	int PixelSize;
	int DilationRadius;
	int DiscretizationScale;

	LogPrinter logPrinter;
};

inline void TrackingValidation::InitDataReader(const std::string validationFileName)
{
	if (fileReader != nullptr)
	{
		delete fileReader;
		fileReader = nullptr;
	}
	fileReader = new BinaryFileReader(Width, Height, PixelSize, validationFileName);
	fileReader->ReadBinaryFileToHostMemory();
}

inline bool TrackingValidation::CheckFileReader() const
{
	if (fileReader == NULL)
	{
		logPrinter.PrintLogs("File Reader is Not Ready!", Error);
		system("Pause");
		return true;
	}
	return false;
}

inline void TrackingValidation::VailidationAll()
{
	if (CheckFileReader()) return;

	auto monitor1 = new Monitor(Width, Height, DilationRadius, DiscretizationScale);
	this->monitor = monitor1;

	const auto frameCount = fileReader->GetFrameCount();
	auto dataPoint = fileReader->GetDataPoint();

	logPrinter.PrintLogs("Test the tracking effect of detect result ... ", Info);
	char iterationText[200];

	DetectResultSegment result;

	for (unsigned i = 0; i<frameCount; ++i)
	{
		sprintf_s(iterationText, 200, "Checking for frame %04d ...", i);
		logPrinter.PrintLogs(iterationText, Info);

		this->monitor->Process(dataPoint[i], &result);
//		CheckPerf(detector->DetectTargets(dataPoint[i], &result), "whole");

		ShowFrame::DrawRectangles(dataPoint[i], &result, Width, Height, 1);
	}
}
