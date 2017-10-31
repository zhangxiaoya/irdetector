#ifndef __MONITOR_H__
#define __MONITOR_H__
#include "../Models/Confidences.hpp"
#include "../Detector/Detector.hpp"
#include "Tracker.hpp"
#include "../Models/DetectResultSegment.hpp"

const int ConfValue = 6;
const int  IncrementConfValue = 12;

const int MaxTrackerCount = 10;
int TrackConfirmThreshold = 30;

class Monitor
{
public:
	Monitor(const int width, const int height, const int dilationRadius, const int discretizationScale);

	~Monitor();

	bool Process(unsigned short* frame);

private:
	void IncreaseConfidenceValueAndUpdateConfidenceQueue() const;

	void DecreaseConfidenceValueMap() const;

	void ResetCurrentDetectMask() const;

	int CalculateQueueSum(int blockX, int blockY) const;

	void GetBlockPos(const TargetPosition& target, int& br, int& bc) const;

	int CheckDistance(const TargetPosition& targetPos, const TargetPosition& trackerPos) const;

	void UpdateTracker(Tracker& tracker, const TargetPosition& targetPos);

	void AddTracker(const TargetPosition& targetPos);

	void UpdateTrackerOrAddTracker(int blockX, int blockY);

protected:
	void InitMonitor();

	void InitDetector();

	void InitConfidenceValueMap();

	void InitTrackerList();

	void ReleaseConfidenceValueMap();

	void ReleaseTrackerList();

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
	DetectResultSegment detectResult;
	DetectResult result;

	Tracker* TrackerList;
};

inline Monitor::Monitor(const int width, const int height, const int dilationRadius, const int discretizationScale) :
	currentFrame(NULL),
	Width(width),
	Height(height),
	DilationRadius(dilationRadius),
	DiscretizationScale(discretizationScale),
	confidences(NULL),
	detector(NULL)
{
	InitMonitor();

	InitDetector();

	InitConfidenceValueMap();

	InitTrackerList();
}

inline Monitor::~Monitor()
{
	delete detector;
	delete confidences;

	ReleaseConfidenceValueMap();

	ReleaseTrackerList();
}

inline void Monitor::IncreaseConfidenceValueAndUpdateConfidenceQueue() const
{
	for (auto i = 0; i < detectResult.targetCount; ++i)
	{
		// Update ConfidenceMap Queue
		const auto centerX = detectResult.targets[i].bottomRightX + detectResult.targets[i].topLeftX;
		const auto centerY = detectResult.targets[i].bottomRightY + detectResult.targets[i].topLeftY;
		const auto BlockX = centerX / BlockSize;
		const auto BlockY = centerY / BlockSize;

		CurrentDetectMask[BlockY * BlockCols + BlockX] = true;
		confidences->ConfidenceMap[BlockY * BlockCols + BlockX][confidences->QueueEnd] = ConfValue;

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

inline int Monitor::CalculateQueueSum(const int blockX, const int blockY) const
{
	auto sum = 0;
	auto i = confidences->QueueBeg;
	while(i != confidences->QueueEnd)
	{
		sum += confidences->ConfidenceMap[blockY * BlockCols + blockX][i];
		i++;
		if (i == CONFIDENCE_QUEUE_ELEM_SIZE)
		{
			i = 0;
		}
	}
	return sum;
}

inline void Monitor::GetBlockPos(const TargetPosition& target, int& br, int& bc) const
{
	const auto centerX = (target.bottomRightX + target.topLeftX) / 2;
	const auto centerY = (target.bottomRightY + target.topLeftY) / 2;
	br = centerY / BlockSize;
	bc = centerX / BlockSize;
}

inline int Monitor::CheckDistance(const TargetPosition& targetPos, const TargetPosition& trackerPos) const
{
	const auto targetCenterX = (targetPos.bottomRightX + targetPos.topLeftX) / 2;
	const auto targetCenterY = (targetPos.bottomRightY + targetPos.topLeftY) / 2;
	const auto trackerCenterX = (trackerPos.bottomRightX + trackerPos.topLeftX) / 2;
	const auto trackerCenterY = (trackerPos.bottomRightY + trackerPos.topLeftY) / 2;

	const auto ManhattanDistance = abs(targetCenterX - trackerCenterX) + abs(targetCenterY - trackerCenterY);
	return  ManhattanDistance;
}

inline void Monitor::UpdateTracker(Tracker& tracker, const TargetPosition& targetPos)
{
	tracker.Postion.bottomRightY = targetPos.bottomRightY;
	tracker.Postion.bottomRightX = targetPos.bottomRightX;
	tracker.Postion.topLeftX = targetPos.topLeftX;
	tracker.Postion.topLeftY = targetPos.topLeftY;
}

inline void Monitor::AddTracker(const TargetPosition& targetPos)
{
	for(auto i = 0; i < MaxTrackerCount; ++i)
	{
		if (TrackerList[i].ValidFlag == true)
		{
			int BC = 0;
			int BR = 0;
			GetBlockPos(targetPos, BR, BC);

			TrackerList[i].Postion.bottomRightY = targetPos.bottomRightY;
			TrackerList[i].Postion.bottomRightX = targetPos.bottomRightX;
			TrackerList[i].Postion.topLeftX = targetPos.topLeftX;
			TrackerList[i].Postion.topLeftY = targetPos.topLeftY;
			TrackerList[i].BlockX = BC;
			TrackerList[i].BlockY = BR;
			break;
		}
	}
}

inline void Monitor::UpdateTrackerOrAddTracker(const int blockX, const int blockY)
{
	auto hasTargetNotTracked = false;

	// Go through detect result and check if there are target in current block without tracked now
	// Without check multi target in one block
	for (auto i = 0; i < result.result->targetCount; ++i)
	{
		if(result.hasTracker[i] == true)
			continue;

		auto BR = 0;
		auto BC = 0;
		GetBlockPos(result.result->targets[i], BR, BC);
		if(blockX == BC && blockY == BR)
		{
			hasTargetNotTracked = true;
			break;
		}
	}

	if(hasTargetNotTracked == true) // if target in this block without tracked is exist
	{
		auto hasTrackerForThisBlock = false;

		for (auto j = 0; j < MaxTrackerCount; ++j)
		{
			if (TrackerList[j].ValidFlag == false)
				continue;
			if(TrackerList[j].BlockX != blockX || TrackerList[j].BlockY != blockY)
				continue;

			// tracker for this block exist
			hasTrackerForThisBlock = true;
			for (auto i = 0; i < result.result->targetCount; ++ i)
			{
				if(result.hasTracker[i] == true)
					continue;

				auto BR = 0;
				auto BC = 0;
				GetBlockPos(result.result->targets[i], BR, BC);
				if (TrackerList[j].BlockX == BC && TrackerList[j].BlockY == BR)
				{
					// Use Manhattan Distance to check if this is the same target with tracker
					if (CheckDistance(result.result->targets[i], TrackerList[j].Postion) < 8)
						UpdateTracker(TrackerList[j], result.result->targets[i]);
					else
						AddTracker(result.result->targets[i]);
					result.hasTracker[i] = true;
				}
			}
		}

		// if there are no tracker
		if(hasTrackerForThisBlock == false)
		{
			for (auto i = 0; i < result.result->targetCount; ++i)
			{
				AddTracker(result.result->targets[i]);
			}
		}
	}
	else
	{
		for(auto j =0 ; j < MaxTrackerCount; ++j)
		{
			if (TrackerList[j].ValidFlag == false)
				continue;
			if(TrackerList[j].BlockX != blockX || TrackerList[j].BlockY != blockY)
				continue;
			// To-Do
			// Re-search
		}
	}
}

inline bool Monitor::Process(unsigned short* frame)
{
	detector->DetectTargets(frame, &detectResult);

	// store current detected targets
	result.result = &detectResult;
	memset(result.hasTracker, false, sizeof(bool) * 5);

	ResetCurrentDetectMask();

	IncreaseConfidenceValueAndUpdateConfidenceQueue();

	DecreaseConfidenceValueMap();

	for (auto R = 0; R < BlockRows; ++R)
	{
		for (auto C = 0; C < BlockCols; ++C)
		{
			const auto TotalConfValue = this->ConfidenceValueMap[R * BlockCols + C] + CalculateQueueSum(C, R);
			if(TotalConfValue > TrackConfirmThreshold)
			{
				UpdateTrackerOrAddTracker(C, R);
			}
		}
	}

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
	memset(this->CurrentDetectMask, false, sizeof(bool) * BlockRows * BlockCols);

	this->CurrentDetectMask = new bool[BlockRows * BlockCols];
	memset(this->ConfidenceValueMap, 0, sizeof(int) * BlockCols * BlockRows);
}

inline void Monitor::InitTrackerList()
{
	this->TrackerList = new Tracker[MaxTrackerCount];
}

inline void Monitor::ReleaseConfidenceValueMap()
{
	delete[] this->ConfidenceValueMap;
	this->ConfidenceValueMap = NULL;

	delete[] this->CurrentDetectMask;
	this->CurrentDetectMask = NULL;
}

inline void Monitor::ReleaseTrackerList()
{
	delete[] TrackerList;
	this->TrackerList = NULL;
}

#endif
