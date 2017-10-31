#ifndef __MONITOR_H__
#define __MONITOR_H__
#include "../Models/Confidences.hpp"
#include "../Detector/Detector.hpp"

const int ConfValue = 6;
const int  IncrementConfValue = 12;

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

private:
	void IncreaseConfidenceValueAndUpdateConfidenceQueue() const;

	void DecreaseConfidenceValueMap() const;

	void ResetCurrentDetectMask() const;

protected:
	void InitMonitor();

	void InitDetector();

	void InitConfidenceValueMap();

	void ReleaseConfidenceValueMap();

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

inline void Monitor::IncreaseConfidenceValueAndUpdateConfidenceQueue() const
{
	for (auto i = 0; i < result.targetCount; ++i)
	{
		// Update ConfidenceMap Queue
		const auto centerX = result.targets[i].bottomRightX + result.targets[i].topLeftX;
		const auto centerY = result.targets[i].bottomRightY + result.targets[i].topLeftY;
		const auto BlockX = centerX / BlockSize;
		const auto BlockY = centerY / BlockSize;

		CurrentDetectMask[BlockY * BlockCols + BlockX] = true;
		confidences->ConfidenceMap[BlockY][BlockX][confidences->QueueEnd] = ConfValue;

		// CinfidenceValueMap Increase
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

	confidences->QueueEnd++;
	if (confidences->QueueBeg == confidences->QueueEnd)
		confidences->QueueBeg++;
}

inline void Monitor::DecreaseConfidenceValueMap() const
{
	// ConfidenceValueMap Decrease
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
}

inline void Monitor::ResetCurrentDetectMask() const
{
	memset(this->CurrentDetectMask, false, sizeof(bool) * BlockRows * BlockCols);
}

inline bool Monitor::Process(unsigned short* frame)
{
	detector->DetectTargets(frame, &result);

	ResetCurrentDetectMask();

	IncreaseConfidenceValueAndUpdateConfidenceQueue();

	DecreaseConfidenceValueMap();

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
	this->detector->SetRemoveFalseAlarmParameters(true, false, false, false, true, true);
}

inline void Monitor::InitConfidenceValueMap()
{
	this->ConfidenceValueMap = new int[BlockRows * BlockCols];
	this->CurrentDetectMask = new bool[BlockRows * BlockCols];
	memset(this->CurrentDetectMask, false, sizeof(bool) * BlockRows * BlockCols);
	memset(this->ConfidenceValueMap, 0, sizeof(int) * BlockCols * BlockRows);
}

inline void Monitor::ReleaseConfidenceValueMap()
{
	delete[] this->ConfidenceValueMap;
	this->ConfidenceValueMap = NULL;

	delete[] this->CurrentDetectMask;
	this->CurrentDetectMask = NULL;
}

#endif
