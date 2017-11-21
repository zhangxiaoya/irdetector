#ifndef __MONITOR_H__
#define __MONITOR_H__
#include "../Models/Confidences.hpp"
#include "../Detector/Detector.hpp"
#include "Tracker.hpp"
#include "../Models/DetectResultSegment.hpp"

bool IsTracking = true; // For debug tracking

const int ConfValue = 6; // Confidence value for queue
const int IncrementConfValue = 12; // Confidence value for map

const int MaxTrackerCount = 10; // Max tracker count
const int TrackConfirmThreshold = 12; // Confirm tracking target threshold
const int MaxConfidenceValue = 100; // Max confidence value for per block

// Monitor class
class Monitor
{
public:
	Monitor(const int width, const int height, const int dilationRadius, const int discretizationScale);

	~Monitor();

	// Main process method
	bool Process(unsigned short* frame, DetectResultSegment* result);

private:
	void UpdateConfidenceValueAndUpdateConfidenceQueue() const;

	void DecreaseConfidenceValueMap() const;

	void ResetCurrentDetectMask() const;

	int CalculateQueueSum(int blockX, int blockY) const;

	void GetBlockPos(const TargetPosition& target, int& br, int& bc) const;

	int CheckDistance(const TargetPosition& targetPos, const TargetPosition& trackerPos) const;

	void UpdateTracker(Tracker& tracker, const TargetPosition& targetPos, bool isExtendLifetime = true);

	void AddTracker(const TargetPosition& targetPos);

	void UpdateTrackerOrAddTrackerForBlockUnit(int blockX, int blockY);

	void UpdateTrackerForAllBlocks(unsigned short* frame);

	double GetContrastRate(unsigned short* frame, int left, int top, int width, int height);

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
		const auto centerX = (detectResult.targets[i].bottomRightX + detectResult.targets[i].topLeftX) / 2;
		const auto centerY = (detectResult.targets[i].bottomRightY + detectResult.targets[i].topLeftY) / 2;
		const auto BlockX = centerX / BlockSize;
		const auto BlockY = centerY / BlockSize;

		// Set block mask unit to true
		CurrentDetectMask[BlockY * BlockCols + BlockX] = true;
		// Insert confidence value to this queue
		confidences->ConfidenceMap[BlockY * BlockCols + BlockX][confidences->QueueEnd] = ConfValue;

		// Increase confidence value for confidence value map unit and its neighbor units
		ConfidenceValueMap[BlockY * BlockCols + BlockX] += IncrementConfValue;
		if (ConfidenceValueMap[BlockY * BlockCols + BlockX] > MaxConfidenceValue)
			ConfidenceValueMap[BlockY * BlockCols + BlockX] = MaxConfidenceValue;
		// Left
		if (BlockX - 1 >= 0)
		{
			ConfidenceValueMap[BlockY * BlockCols + BlockX - 1] += IncrementConfValue / 2;
			if (ConfidenceValueMap[BlockY * BlockCols + BlockX - 1] > MaxConfidenceValue)
				ConfidenceValueMap[BlockY * BlockCols + BlockX - 1] = MaxConfidenceValue;
			CurrentDetectMask[BlockY * BlockCols + BlockX - 1] = true;
		}
		// Top
		if (BlockY - 1 >= 0)
		{
			ConfidenceValueMap[(BlockY - 1) * BlockCols + BlockX] += IncrementConfValue / 2;
			if (ConfidenceValueMap[(BlockY - 1) * BlockCols + BlockX] > MaxConfidenceValue)
				ConfidenceValueMap[(BlockY - 1) * BlockCols + BlockX] = MaxConfidenceValue;
			CurrentDetectMask[(BlockY - 1) * BlockCols + BlockX] = true;
		}
		// Right
		if (BlockX + 1 < BlockCols)
		{
			ConfidenceValueMap[BlockY * BlockCols + BlockX + 1] += IncrementConfValue / 2;
			if (ConfidenceValueMap[BlockY * BlockCols + BlockX + 1] > MaxConfidenceValue)
				ConfidenceValueMap[BlockY * BlockCols + BlockX + 1] = MaxConfidenceValue;
			CurrentDetectMask[BlockY * BlockCols + BlockX + 1] = true;
		}
		// Bottom
		if (BlockY + 1 < BlockRows)
		{
			ConfidenceValueMap[(BlockY + 1) * BlockCols + BlockX] += IncrementConfValue / 2;
			if (ConfidenceValueMap[(BlockY + 1) * BlockCols + BlockX] > MaxConfidenceValue)
				ConfidenceValueMap[(BlockY + 1) * BlockCols + BlockX] = MaxConfidenceValue;
			CurrentDetectMask[(BlockY + 1) * BlockCols + BlockX] = true;
		}

		// Top-Left
		if (BlockX - 1 >= 0 && BlockY - 1 >= 0)
		{
			ConfidenceValueMap[(BlockY - 1) * BlockCols + BlockX - 1] += IncrementConfValue / 2;
			if (ConfidenceValueMap[(BlockY - 1) * BlockCols + BlockX - 1] > MaxConfidenceValue)
				ConfidenceValueMap[(BlockY - 1) * BlockCols + BlockX - 1] = MaxConfidenceValue;
			CurrentDetectMask[(BlockY - 1) * BlockCols + BlockX - 1] = true;
		}
		// Top-Right
		if (BlockX + 1 < BlockCols && BlockY - 1 >= 0)
		{
			ConfidenceValueMap[(BlockY - 1) * BlockCols + BlockX + 1] += IncrementConfValue / 2;
			if (ConfidenceValueMap[(BlockY - 1) * BlockCols + BlockX + 1] > MaxConfidenceValue)
				ConfidenceValueMap[(BlockY - 1) * BlockCols + BlockX + 1] = MaxConfidenceValue;
			CurrentDetectMask[(BlockY - 1) * BlockCols + BlockX + 1] = true;
		}
		// Bottom-Left
		if (BlockX - 1 >= 0 && BlockY + 1 < BlockRows)
		{
			ConfidenceValueMap[(BlockY + 1) * BlockCols + BlockX - 1] += IncrementConfValue / 2;
			if (ConfidenceValueMap[(BlockY + 1) * BlockCols + BlockX - 1] > MaxConfidenceValue)
				ConfidenceValueMap[(BlockY + 1) * BlockCols + BlockX - 1] = MaxConfidenceValue;
			CurrentDetectMask[(BlockY + 1) * BlockCols + BlockX - 1] = true;
		}
		// Bottom-Right
		if (BlockX + 1 < BlockCols && BlockY + 1 < BlockRows)
		{
			ConfidenceValueMap[(BlockY + 1) * BlockCols + BlockX + 1] += IncrementConfValue / 2;
			if (ConfidenceValueMap[(BlockY + 1) * BlockCols + BlockX + 1] > MaxConfidenceValue)
				ConfidenceValueMap[(BlockY + 1) * BlockCols + BlockX + 1] = MaxConfidenceValue;
			CurrentDetectMask[(BlockY + 1) * BlockCols + BlockX + 1] = true;
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
	while (i != confidences->QueueEnd)
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
	return ManhattanDistance;
}

inline void Monitor::UpdateTracker(Tracker& tracker, const TargetPosition& targetPos, bool isExtendLifetime)
{
	tracker.Postion.bottomRightY = targetPos.bottomRightY;
	tracker.Postion.bottomRightX = targetPos.bottomRightX;
	tracker.Postion.topLeftX = targetPos.topLeftX;
	tracker.Postion.topLeftY = targetPos.topLeftY;

	int bc = 0;
	int br = 0;
	GetBlockPos(targetPos, br, bc);
	tracker.BlockX = bc;
	tracker.BlockY = br;
	if(isExtendLifetime == true)
		tracker.ExtendLifeTime();
}

inline void Monitor::AddTracker(const TargetPosition& targetPos)
{
	for (auto i = 0; i < MaxTrackerCount; ++i)
	{
		if (TrackerList[i].ValidFlag == false)
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
			TrackerList[i].ValidFlag = true;
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
		if (detectResultWithStatus.hasTracker[targetIdx] == true)
			continue;

		auto BR = 0;
		auto BC = 0;
		GetBlockPos(detectResultWithStatus.detectResultPointers->targets[targetIdx], BR, BC);
		if (blockX == BC && blockY == BR)
		{
			hasTargetNotTracked = true;
			break;
		}
	}

	if (hasTargetNotTracked == true) // if target in this block without tracked is exist
	{
		auto hasTrackerForThisBlock = false;

		for (auto trackerIdx = 0; trackerIdx < MaxTrackerCount; ++trackerIdx)
		{
			if (TrackerList[trackerIdx].ValidFlag == false)
				continue;
			if (TrackerList[trackerIdx].BlockX != blockX || TrackerList[trackerIdx].BlockY != blockY)
				continue;

			// tracker for this block exist
			hasTrackerForThisBlock = true;
			for (auto targetIdx = 0; targetIdx < detectResultWithStatus.detectResultPointers->targetCount; ++targetIdx)
			{
				if (detectResultWithStatus.hasTracker[targetIdx] == true)
					continue;

				auto BR = 0;
				auto BC = 0;
				// To-Do need check
				GetBlockPos(detectResultWithStatus.detectResultPointers->targets[targetIdx], BR, BC);
				if ((TrackerList[trackerIdx].BlockX == BC && TrackerList[trackerIdx].BlockY == BR)
					|| (BC - 1 >= 0 && TrackerList[trackerIdx].BlockX == BC - 1 && TrackerList[trackerIdx].BlockY == BR)
					|| (BR - 1 >= 0 && TrackerList[trackerIdx].BlockX == BC && TrackerList[trackerIdx].BlockY == BR - 1)
					|| (BC + 1 < BlockCols && TrackerList[trackerIdx].BlockX == BC + 1 && TrackerList[trackerIdx].BlockY == BR)
					|| (BR + 1 < BlockRows && TrackerList[trackerIdx].BlockX == BC && TrackerList[trackerIdx].BlockY == BR + 1)
					|| (BC - 1 >= 0 && BR - 1 >= 0 && TrackerList[trackerIdx].BlockX == BC - 1 && TrackerList[trackerIdx].BlockY == BR - 1)
					|| (BC + 1 < BlockCols && BR - 1 >= 0 && TrackerList[trackerIdx].BlockX == BC + 1 && TrackerList[trackerIdx].BlockY == BR - 1)
					|| (BC + 1 < BlockCols && BR + 1 < BlockRows && TrackerList[trackerIdx].BlockX == BC + 1 && TrackerList[trackerIdx].BlockY == BR + 1)
					|| (BC - 1 >= 0 && BR + 1 < BlockRows && TrackerList[trackerIdx].BlockX == BC - 1 && TrackerList[trackerIdx].BlockY == BR + 1)
				)
				{
					// Use Manhattan Distance to check if this is the same target with tracker
					if (CheckDistance(detectResultWithStatus.detectResultPointers->targets[targetIdx], TrackerList[trackerIdx].Postion) < 10)
						UpdateTracker(TrackerList[trackerIdx], detectResultWithStatus.detectResultPointers->targets[targetIdx]);

					else
						AddTracker(detectResultWithStatus.detectResultPointers->targets[targetIdx]);

					detectResultWithStatus.hasTracker[targetIdx] = true;
				}
			}
		}

		// if there are no tracker
		if (hasTrackerForThisBlock == false)
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

inline void Monitor::UpdateTrackerForAllBlocks(unsigned short* frame)
{
	// Step 1. check if has tracker

	// Step 2. if has tracker, then update all tracker for their target

	// Step 3. if there no tracker, or there are no target has been tracked, then add tracker for them

	// This is step 1;
	bool hasTracker = false;
	for (int i = 0; i < MaxTrackerCount; ++i)
	{
		if (TrackerList[i].ValidFlag == true)
		{
			hasTracker = true;
			break;
		}
	}

	// This is step 2
	if (hasTracker == true)
	{
		for (int i = 0; i < MaxTrackerCount; ++i)
		{
			if (TrackerList[i].ValidFlag == true)
			{
				// Step first: find search region base this tracker
				auto centerX = (TrackerList[i].Postion.bottomRightX + TrackerList[i].Postion.topLeftX) / 2;
				auto centerY = (TrackerList[i].Postion.bottomRightY + TrackerList[i].Postion.topLeftY) / 2;

				auto searchRegionLeft = centerX - BlockSize;
				searchRegionLeft = searchRegionLeft < 0 ? 0 : searchRegionLeft;
				auto searchRegionTop = centerY - BlockSize;
				searchRegionTop = searchRegionTop < 0 ? 0 : searchRegionTop;
				auto searchRegionRight = centerX + BlockSize;
				searchRegionRight = searchRegionRight < Width ? searchRegionRight : Width;
				auto searchRegionBottom = centerY + BlockSize;
				searchRegionBottom = searchRegionBottom < Height ? searchRegionBottom : Height;

				bool updateTrackerInfoSuccess = false;
				for (int j = 0; j < detectResultWithStatus.detectResultPointers->targetCount; ++j)
				{
					if (detectResultWithStatus.hasTracker[j] == true)
						continue;
					auto targetCenterX = (detectResultWithStatus.detectResultPointers->targets[j].bottomRightX + detectResultWithStatus.detectResultPointers->targets[j].topLeftX) / 2;
					auto targetCenterY = (detectResultWithStatus.detectResultPointers->targets[j].bottomRightY + detectResultWithStatus.detectResultPointers->targets[j].topLeftY) / 2;
					// if find one target in this search region, then update this tracker info, actually need find all target int this region, then the nearest target should be last result
					// this is simple way, need to-do
					if (targetCenterX > searchRegionLeft && targetCenterX < searchRegionRight && targetCenterY > searchRegionTop && targetCenterY < searchRegionBottom)
					{
						updateTrackerInfoSuccess = true;
						UpdateTracker(TrackerList[i], detectResultWithStatus.detectResultPointers->targets[j]);
						detectResultWithStatus.hasTracker[j] = true;
					}
				}
				// if tracker cannot find one target in it's search region, then shrink it's lifetime
				// this is simple way, if there no target detect in this region, need tracker research target
				// need to-do
				if (updateTrackerInfoSuccess == false)
				{
					auto targetWidth = TrackerList[i].Postion.bottomRightX - TrackerList[i].Postion.topLeftX + 1;
					auto targetHeight = TrackerList[i].Postion.bottomRightY - TrackerList[i].Postion.topLeftY + 1;

					double maxContrast = 0.0;
					double maxR = searchRegionTop;
					double maxC = searchRegionLeft;
					for (int r = searchRegionTop; r <= searchRegionBottom - targetHeight; ++r)
					{
						for (int c = searchRegionLeft; c <= searchRegionRight - targetWidth; ++c)
						{
							double currentContrast = GetContrastRate(frame, c, r, targetWidth, targetHeight);
							if (currentContrast > maxContrast)
							{
								maxContrast = currentContrast;
								maxR = r;
								maxC = c;
							}
						}
					}

					auto researchSuccessFlag = maxContrast > DiscretizationScale * 4;
					if (researchSuccessFlag == false)
					{
						TrackerList[i].ShrinkLifeTime();
					}
					else
					{
						TargetPosition pos;
						pos.topLeftX = maxC;
						pos.topLeftY = maxR;
						pos.bottomRightX = maxC + targetWidth;
						pos.bottomRightY = maxR + targetHeight;
						UpdateTracker(TrackerList[i], pos, false);
					}
				}
			}
		}
	}
	// This is step 3 : Check all detect targets
	for (int i = 0; i < detectResultWithStatus.detectResultPointers->targetCount; ++i)
	{
		// if this detected target already has tracker for it, do nothing
		if (detectResultWithStatus.hasTracker[i] == true)
			continue;

		// if no tracker for this target, calculate it's block postion
		int br = 0;
		int bc = 0;
		GetBlockPos(detectResultWithStatus.detectResultPointers->targets[i], br, bc);

		auto TotalConfValue = this->ConfidenceValueMap[br * BlockCols + bc] + CalculateQueueSum(bc, br);
		if (TotalConfValue >= TrackConfirmThreshold)
		{
			AddTracker(detectResultWithStatus.detectResultPointers->targets[i]);
			detectResultWithStatus.hasTracker[i] = true;
		}
	}
}

inline double Monitor::GetContrastRate(unsigned short* frame, int left, int top, int width, int height)
{
	double result = 0.0;
	int widthPadding = width;
	int heightPadding = height;

	double avgTarget = 0.0;
	double avgSurrouding = 0.0;
	double maxTarget = 0.0;

	// target max value
	for (int r = top; r < top + height; ++r)
	{
		for (int c = left; c < left + width; ++c)
		{
			if (maxTarget < (double)frame[r * Width + c])
				maxTarget = (double)frame[r * Width + c];
		}
	}

	// target average gray value
	double sum = 0.0;
	for (int r = top; r < top + height; ++r)
	{
		double sumRow = 0.0;
		for (int c = left; c < left + width; ++c)
		{
			sumRow += (double)frame[r * Width + c];
		}
		sum += (sumRow / width);
	}
	avgTarget = sum / height;

	// target surrounding average gray value
	sum = 0.0;
	int surroundingTop = top - heightPadding;
	surroundingTop = surroundingTop < 0 ? 0 : surroundingTop;
	int surroundingLeft = left - widthPadding;
	surroundingLeft = surroundingLeft < 0 ? 0 : surroundingLeft;
	int surroundingRight = left + width + widthPadding;
	surroundingRight = surroundingRight > Width ? Width : surroundingRight;
	int surroundingBottom = top + height + heightPadding;
	surroundingBottom = surroundingBottom > Height ? Height : surroundingBottom;
	for (int r = surroundingTop; r < top; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += (double)frame[r * Width + c];
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (int r = top + height; r < surroundingBottom; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += (double)frame[r * Width + c];
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (int r = top; r < top + height; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < left; ++c)
		{
			sumRow += (double)frame[r * Width + c];
		}
		for (int c = left + width; c < surroundingRight; ++c)
		{
			sumRow += (double)frame[r * Width + c];
		}
		sum += sumRow / ((left - surroundingLeft) + (surroundingRight - (left + width)));
	}

	// for (int r = surroundingTop; r < surroundingBottom; ++r)
	// {
	// 	double sumRow = 0.0;
	// 	for (int c = surroundingLeft; c < surroundingRight; ++c)
	// 	{
	// 		sumRow += (double)frame[r * Width + c];
	// 	}
	// 	sum += (sumRow / (surroundingRight - surroundingLeft));
	// }
	avgSurrouding = sum / (surroundingBottom - surroundingTop);

	// result = maxTarget / avgSurrouding;
	result = maxTarget - avgSurrouding;
	return result;
}

/******************************************************************************
/*
/*  Main Process Method
/*
******************************************************************************/
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
	UpdateTrackerForAllBlocks(frame);

	// copy tracking result back
	memcpy(result->header, detectResult.header, 16);
	result->targetCount = detectResult.targetCount;
	if (IsTracking == false)
	{
		memcpy(result->targets, detectResult.targets, sizeof(TargetPosition) * 5);
	}
	else
	{
		int trackingTargetCount = 0;
		for (auto i = 0; i < MaxTrackerCount; ++i)
		{
			if (TrackerList[i].ValidFlag == true && TrackerList[i].LifeTime > 2)
			{
				result->targets[trackingTargetCount] = TrackerList[i].Postion;
				trackingTargetCount++;
				if (trackingTargetCount >= 5)
					break;
			}
		}
		result->targetCount = trackingTargetCount;
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
	this->detector->InitSpace();
	this->detector->SetRemoveFalseAlarmParameters(false, false, false, false, true, true);
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
	this->ConfidenceValueMap = nullptr;

	delete[] this->CurrentDetectMask;
	this->CurrentDetectMask = nullptr;
}

inline void Monitor::ReleaseTrackerList()
{
	delete[] TrackerList;
	this->TrackerList = nullptr;
}

#endif
