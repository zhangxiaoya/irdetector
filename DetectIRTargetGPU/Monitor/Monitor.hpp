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

	bool Process(unsigned short* frame, DetectResultSegment* result);

private:
	void UpdateConfidenceValueAndUpdateConfidenceQueue() const;

	void DecreaseConfidenceValueMap() const;

	void ResetCurrentDetectMask() const;

	int CalculateQueueSum(int blockX, int blockY) const;

	void GetBlockPos(const TargetPosition& target, int& br, int& bc) const;

	int CheckDistance(const TargetPosition& targetPos, const TargetPosition& trackerPos) const;

	void UpdateTracker(Tracker& tracker, const TargetPosition& targetPos);

	void AddTracker(const TargetPosition& targetPos);

	void UpdateTrackerOrAddTrackerForBlockUnit(int blockX, int blockY);

	void UpdateTrackerForAllBlocks();

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
	DetectResultWithTrackerStatus detectResultWithStatus;

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

inline void Monitor::UpdateConfidenceValueAndUpdateConfidenceQueue() const
{
	// all queue unit should be zero
	for (auto R = 0; R < BlockRows; ++R)
	{
		for (auto C = 0; C < BlockCols; ++C)
		{
			confidences->ConfidenceMap[R * BlockCols + C][confidences->QueueEnd] = 0;
		}
	}

	for (auto i = 0; i < detectResult.targetCount; ++i)
	{
		// Get block coordination
		const auto centerX = detectResult.targets[i].bottomRightX + detectResult.targets[i].topLeftX;
		const auto centerY = detectResult.targets[i].bottomRightY + detectResult.targets[i].topLeftY;
		const auto BlockX = centerX / BlockSize;
		const auto BlockY = centerY / BlockSize;

		// Set block mask unit to true
		CurrentDetectMask[BlockY * BlockCols + BlockX] = true;
		// Insert confidence value to this queue
		confidences->ConfidenceMap[BlockY * BlockCols + BlockX][confidences->QueueEnd] = ConfValue;

		// Increase confidence value for confidence value map unit and its neighbor units
		ConfidenceValueMap[BlockY * BlockCols + BlockX] += IncrementConfValue;
		if (BlockX - 1 >= 0)
		{
			ConfidenceValueMap[BlockY * BlockCols + BlockX - 1] += IncrementConfValue / 2;
			CurrentDetectMask[BlockY * BlockCols + BlockX - 1] = true;
		}
		if (BlockY - 1 >= 0)
		{
			ConfidenceValueMap[(BlockY - 1) * BlockCols + BlockX] += IncrementConfValue / 2;
			CurrentDetectMask[(BlockY - 1) * BlockCols + BlockX] = true;
		}
		if (BlockX + 1 < BlockCols)
		{
			ConfidenceValueMap[BlockY * BlockCols + BlockX + 1] += IncrementConfValue / 2;
			CurrentDetectMask[BlockY * BlockCols + BlockX + 1] = true;

		}
		if (BlockY + 1 < BlockRows)
		{
			ConfidenceValueMap[(BlockY + 1) * BlockCols + BlockX] += IncrementConfValue / 2;
			CurrentDetectMask[(BlockY + 1) * BlockCols + BlockX] = true;
		}
	}

	// update confidence queue front and end index
	confidences->QueueEnd++;
	if (confidences->QueueEnd >= CONFIDENCE_QUEUE_ELEM_SIZE)
		confidences->QueueEnd = 0;
	if (confidences->QueueBeg == confidences->QueueEnd)
	{
		confidences->QueueBeg++;
		if (confidences->QueueBeg >= CONFIDENCE_QUEUE_ELEM_SIZE)
			confidences->QueueBeg = 0;
	}

	// if not detect targets in this block, shrink confidence value
	DecreaseConfidenceValueMap();
}

inline void Monitor::DecreaseConfidenceValueMap() const
{
	for (auto R = 0; R < BlockRows; ++R)
	{
		for (auto C = 0; C < BlockCols; ++C)
		{
			if (CurrentDetectMask[R * BlockCols + C] == false)
				ConfidenceValueMap[R * BlockCols + C] -= IncrementConfValue;
			if (ConfidenceValueMap[R * BlockCols + C] < 0)
				ConfidenceValueMap[R * BlockCols + C] = 0;
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

			TrackerList[i].InitLifeTime();
			break;
		}
	}
}

inline void Monitor::UpdateTrackerOrAddTrackerForBlockUnit(const int blockX, const int blockY)
{
	// flag if has deteted target without tracked
	auto hasTargetNotTracked = false;

	// Go through detect result and check if there are target in current block without tracked now
	for (auto targetIdx = 0; targetIdx < detectResultWithStatus.detectResultPointers->targetCount; ++targetIdx)
	{
		if(detectResultWithStatus.hasTracker[targetIdx] == true)
			continue;

		auto BR = 0;
		auto BC = 0;
		GetBlockPos(detectResultWithStatus.detectResultPointers->targets[targetIdx], BR, BC);
		if(blockX == BC && blockY == BR)
		{
			hasTargetNotTracked = true;
			break;
		}
	}

	if(hasTargetNotTracked == true) // if target in this block without tracked is exist
	{
		auto hasTrackerForThisBlock = false;

		for (auto trackerIdx = 0; trackerIdx < MaxTrackerCount; ++trackerIdx)
		{
			if (TrackerList[trackerIdx].ValidFlag == false)
				continue;
			if(TrackerList[trackerIdx].BlockX != blockX || TrackerList[trackerIdx].BlockY != blockY)
				continue;

			// tracker for this block exist
			hasTrackerForThisBlock = true;
			for (auto targetIdx = 0; targetIdx < detectResultWithStatus.detectResultPointers->targetCount; ++ targetIdx)
			{
				if(detectResultWithStatus.hasTracker[targetIdx] == true)
					continue;

				auto BR = 0;
				auto BC = 0;
				// To-Do need check
				GetBlockPos(detectResultWithStatus.detectResultPointers->targets[targetIdx], BR, BC);
				if (TrackerList[trackerIdx].BlockX == BC && TrackerList[trackerIdx].BlockY == BR)
				{
					// Use Manhattan Distance to check if this is the same target with tracker
					if (CheckDistance(detectResultWithStatus.detectResultPointers->targets[targetIdx], TrackerList[trackerIdx].Postion) < 8)
						UpdateTracker(TrackerList[trackerIdx], detectResultWithStatus.detectResultPointers->targets[targetIdx]);
					else
						AddTracker(detectResultWithStatus.detectResultPointers->targets[targetIdx]);

					detectResultWithStatus.hasTracker[targetIdx] = true;
				}
			}
		}

		// if there are no tracker
		if(hasTrackerForThisBlock == false)
		{
			for (auto i = 0; i < detectResultWithStatus.detectResultPointers->targetCount; ++i)
			{
				AddTracker(detectResultWithStatus.detectResultPointers->targets[i]);
			}
		}
	}
	else
	{
		for (auto trackerIdx = 0; trackerIdx < MaxTrackerCount; ++trackerIdx)
		{
			if (TrackerList[trackerIdx].ValidFlag == false)
				continue;
			if (TrackerList[trackerIdx].BlockX != blockX || TrackerList[trackerIdx].BlockY != blockY)
				continue;
			// To-Do
			// Re-search
			// Sample way : cannot find target again, to shrink lifetime 
			TrackerList[trackerIdx].ShrinkLifeTime();
		}
	}
}

inline void Monitor::UpdateTrackerForAllBlocks()
{
	for (auto R = 0; R < BlockRows; ++R)
	{
		for (auto C = 0; C < BlockCols; ++C)
		{
			// Get Total confidence value: block confidence value and sum of confidence queue
			const auto TotalConfValue = this->ConfidenceValueMap[R * BlockCols + C] + CalculateQueueSum(C, R);
			if(TotalConfValue > TrackConfirmThreshold)
			{
				UpdateTrackerOrAddTrackerForBlockUnit(C, R);
			}
		}
	}
}

inline bool Monitor::Process(unsigned short* frame, DetectResultSegment* result)
{
	// detect target in single frame
	detector->DetectTargets(frame, &detectResult);

	// copy detect result and set default tracking status
	detectResultWithStatus.detectResultPointers = &detectResult;
	memset(detectResultWithStatus.hasTracker, false, sizeof(bool) * 5);

	// reset current frame block mask map
	ResetCurrentDetectMask();

	// update confidence value map and confidence queue map
	UpdateConfidenceValueAndUpdateConfidenceQueue();

	// update tracker for all blocks
	UpdateTrackerForAllBlocks();

	memcpy(result->header, detectResult.header, 16);
	int trackingTargetCount = 0;
	for (auto i = 0; i < MaxTrackerCount; ++i)
	{
		if(TrackerList[i].ValidFlag == true)
		{
			result->targets[trackingTargetCount] = TrackerList[i].Postion;
			trackingTargetCount++;
			if(trackingTargetCount >= 5)
				break;
		}
	}
	result->targetCount = trackingTargetCount;

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
	this->detector->InitSpace();
	this->detector->SetRemoveFalseAlarmParameters(true, false, false, false, true, true);
}

inline void Monitor::InitConfidenceValueMap()
{
	this->ConfidenceValueMap = new int[BlockRows * BlockCols];
	this->CurrentDetectMask = new bool[BlockRows * BlockCols];

	memset(this->CurrentDetectMask, false, sizeof(bool) * BlockRows * BlockCols);
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
