#ifndef __MONITOR_H__
#define __MONITOR_H__
#include "../Models/Confidences.hpp"
#include "../Detector/Detector.hpp"

const int ConfValue = 12;
const int  IncrementConfValue = 10;

class Monitor
{
public:
	Monitor(const int width, const int height, const int dilationRadius, const int discretizationScale):
		currentFrame(nullptr),
		Width(width),
		Height(height),
		DilationRadius(dilationRadius),
		DiscretizationScale(discretizationScale),
		confidences(nullptr),
		detector(nullptr)
	{
		InitMonitor();

		InitDetector();

		InitConfidenceValueMap();
	}

	~Monitor()
	{
		delete detector;
		delete confidences;
	}

	bool Process(unsigned short* frame);

protected:
	void InitMonitor();

	void InitDetector();

	void InitConfidenceValueMap();

private:
	unsigned short* currentFrame;
	int Width;
	int Height;
	int DilationRadius;
	int DiscretizationScale;

	int BlockCols;
	int BlockRows;

	int* ConfidenceValueMap;
	bool* CurrentDetectMask;

	Confidences* confidences;
	Detector* detector;
	ResultSegment result;
};

inline bool Monitor::Process(unsigned short* frame)
{
	detector->DetectTargets(frame, &result);

	memset(this->CurrentDetectMask, false, sizeof(bool) * BlockRows * BlockCols);

	for (auto i = 0; i < result.targetCount; ++i)
	{
		const auto centerX = result.targets[i].bottomRightX + result.targets[i].topLeftX;
		const auto centerY = result.targets[i].bottomRightY + result.targets[i].topLeftY;
		const auto BlockX = centerX / BlockSize;
		const auto BlockY = centerY / BlockSize;

		CurrentDetectMask[BlockY * BlockCols + BlockX] = true;
		confidences->ConfidenceMap[BlockY][BlockX][confidences->QueueEnd] = ConfValue;

		ConfidenceValueMap[BlockY * BlockCols + BlockX] += IncrementConfValue;
		if (BlockX - 1 >= 0)
			ConfidenceValueMap[BlockY * BlockCols + BlockX - 1] += IncrementConfValue / 2;
		if (BlockY - 1 >= 0)
			ConfidenceValueMap[(BlockY - 1) * BlockCols + BlockX] += IncrementConfValue / 2;
		if (BlockX + 1 < BlockCols)
			ConfidenceValueMap[BlockY * BlockCols + BlockX + 1] += IncrementConfValue / 2;
		if (BlockY + 1 < BlockRows)
			ConfidenceValueMap[(BlockY + 1) * BlockCols + BlockX] += IncrementConfValue / 2;
	}
	for (auto r = 0; r < BlockRows; ++r)
	{
		for (auto c = 0; c < BlockCols; ++c)
		{
			if (CurrentDetectMask[r * BlockCols + c] == false)
			{
				ConfidenceValueMap[r * BlockCols + c] -= IncrementConfValue;
				if (ConfidenceValueMap[r * BlockCols + c] < 0)
					ConfidenceValueMap[r * BlockCols + c] = 0;
			}
		}
	}

	confidences->QueueEnd++;
	if (confidences->QueueBeg == confidences->QueueEnd)
		confidences->QueueBeg++;

	return true;
}

inline void Monitor::InitMonitor()
{
	BlockCols = (Width + (BlockSize - 1)) / BlockSize;
	BlockRows = (Height + (BlockSize - 1)) / BlockSize;

	this->confidences = new Confidences(Width, Height, BlockCols, BlockRows);
	this->confidences->InitConfidenceMap();
}

inline void Monitor::InitDetector()
{
	this->detector = new Detector(Width, Height, DilationRadius, DiscretizationScale);
}

inline void Monitor::InitConfidenceValueMap()
{
	this->ConfidenceValueMap = new int[BlockRows * BlockCols];
	this->CurrentDetectMask = new bool[BlockRows * BlockCols];
	memset(this->CurrentDetectMask, false, sizeof(bool) * BlockRows * BlockCols);
}

#endif
