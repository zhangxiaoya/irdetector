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
	explicit PerformanceValidation(BinaryFileReader* file_reader = nullptr)
		: fileReader(file_reader),
		  detector(nullptr),
		  width(320),
		  height(256)
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

	int width;
	int height;

	LogPrinter logPrinter;
};

inline void PerformanceValidation::InitDataReader(std::string validationFileName)
{
	if(fileReader != nullptr)
	{
		delete fileReader;
		fileReader = nullptr;
	}
	fileReader = new BinaryFileReader(validationFileName);
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

	this->detector = new Detector();
	detector->InitSpace();
	detector->SetAllParameters();

	auto frameCount = fileReader->GetFrameCount();
	auto dataPoint = fileReader->GetDataPoint();

	logPrinter.PrintLogs("Test the visual effect of detect result ... ", Info);
	char iterationText[200];

	ResultSegment result;
	detector->SetAllParameters();

	for(auto i = 0;i<frameCount;++i)
	{
		sprintf_s(iterationText, 200, "Checking for frame %04d ...", i);
		logPrinter.PrintLogs(iterationText, Info);

		CheckPerf(detector->DetectTargets(dataPoint[i], &result), "whole");

		ShowFrame::DrawRectangles(dataPoint[i], &result, width, height);
	}
}