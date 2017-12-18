#pragma once
#include "../DataReaderFromeFiles/BinaryFileReader.hpp"
#include "../Models/LogLevel.hpp"
#include "../LogPrinter/LogPrinter.hpp"
#include "../Checkers/CheckCUDAReturnStatus.h"
#include "../CCL/MeshCCLOnCPU.hpp"
#include "../Checkers/CheckPerf.h"
#include "../Detector/Detector.hpp"
#include "../Detector/LazyDetector.hpp"

class LazyDetectorValidation
{
public:
	LazyDetectorValidation(const int width,
		const int height,
		const int pixelSize,
		const int dilationRadius,
		const int discretizationScale)
		:Width(width),
		Height(height),
		lazyDetector(NULL),
		DataPoint(NULL),
		PixelSize(pixelSize),
		DilationRadius(dilationRadius),
		DiscretizationScale(discretizationScale)
	{
	}

	~LazyDetectorValidation()
	{
		delete lazyDetector;
		delete[] DataPoint;
	}

	void InitTestData();

	void VailidationAll();

private:
	LazyDetector* lazyDetector;
	unsigned char* DataPoint;

	int Width;
	int Height;
	int PixelSize;
	int DilationRadius;
	int DiscretizationScale;

	LogPrinter logPrinter;
};

inline void LazyDetectorValidation::InitTestData()
{
	this->DataPoint = new unsigned char[Width * Height * PixelSize];
	for (int i = 0; i < Width * Height; ++i)
	{
		((unsigned short*)DataPoint)[i] = unsigned short(2000);
	}
	// 1 * 1
	((unsigned short*)DataPoint)[(Height / 2) * Width + Width / 2] = 3002;
	// 2 * 2
	((unsigned short*)DataPoint)[(Height / 2 + 10) * Width + Width / 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 10) * Width + Width / 2 + 1] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 11) * Width + Width / 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 11) * Width + Width / 2 +1] = 3002;
	// 3 * 3
	((unsigned short*)DataPoint)[(Height / 2 + 20) * Width + Width / 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 20) * Width + Width / 2 + 1] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 20) * Width + Width / 2 + 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 21) * Width + Width / 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 21) * Width + Width / 2 + 1] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 21) * Width + Width / 2 + 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 22) * Width + Width / 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 22) * Width + Width / 2 + 1] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 22) * Width + Width / 2 + 2] = 3002;
	// 4 * 4
	((unsigned short*)DataPoint)[(Height / 2 + 40) * Width + Width / 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 40) * Width + Width / 2 + 1] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 40) * Width + Width / 2 + 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 40) * Width + Width / 2 + 3] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 41) * Width + Width / 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 41) * Width + Width / 2 + 1] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 41) * Width + Width / 2 + 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 41) * Width + Width / 2 + 3] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 42) * Width + Width / 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 42) * Width + Width / 2 + 1] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 42) * Width + Width / 2 + 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 42) * Width + Width / 2 + 3] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 43) * Width + Width / 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 43) * Width + Width / 2 + 1] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 43) * Width + Width / 2 + 2] = 3002;
	((unsigned short*)DataPoint)[(Height / 2 + 43) * Width + Width / 2 + 3] = 3002;
}

inline void LazyDetectorValidation::VailidationAll()
{
	this->lazyDetector = new LazyDetector(Width, Height, DilationRadius, DiscretizationScale);
	this->lazyDetector->InitDetector();

	logPrinter.PrintLogs("Test the tracking effect of detect result ... ", Info);
	char iterationText[200];

	DetectResultSegment result;
	auto delay = 1;

	while (true)
	{
		CheckPerf(this->lazyDetector->DetectTargets((unsigned short*)DataPoint, &result), "whole lazy detect process");

		ShowFrame::DrawRectangles((unsigned short*)DataPoint, &result, Width, Height, delay);

		Sleep(10);
	}
}
