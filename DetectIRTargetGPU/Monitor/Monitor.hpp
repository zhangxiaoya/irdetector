#ifndef __MONITOR_H__
#define __MONITOR_H__
#include "../Models/Confidences.hpp"
#include "../Detector/Detector.hpp"
#include "../Detector/LazyDetector.hpp"
#include "Tracker.hpp"
#include "../Models/DetectResultSegment.hpp"

const int ConfValue = 1;           // Confidence value for queue
const int IncrementConfValue = 1;  // Confidence value for map

const int MaxTrackerCount = 10;      // Max tracker count
const int TrackConfirmThreshold = 6; // Confirm tracking target threshold
const int MaxConfidenceValue = 5;    // Max confidence value for per block

inline bool CompareTracker(Tracker& a, Tracker& b)
{
	return a.ValidFlag && b.ValidFlag && a.LifeTime > b.LifeTime;
}

// Monitor class
class Monitor
{
public:
	Monitor(const int width, const int height, const int dilationRadius, const int discretizationScale);

	~Monitor();

	// Main process method
	bool Process(unsigned short* frame, DetectResultSegment* result);

	void InitDetector();

private:
	void UpdateConfidenceValueAndUpdateConfidenceQueue() const;

	void DecreaseConfidenceValueMap() const;

	void ResetCurrentDetectMask() const;

	int CalculateQueueSum(int blockX, int blockY) const;

	void GetBlockPos(const TargetPosition& target, int& br, int& bc) const;

	int CheckDistance(const TargetPosition& targetPos, const TargetPosition& trackerPos) const;

	void UpdateTracker(Tracker& tracker, const TargetPosition& targetPos, const TargetInfo& targetInfo, bool isExtendLifetime = true);

	void AddTracker(const TargetPosition& targetPos, const TargetInfo& targetInfo);

	void UpdateTrackerForAllBlocks(unsigned short* frame);

	double GetContrastRate(unsigned short* frame, int left, int top, int width, int height);

	int GetArea(unsigned short* frame, int left, int top, int width, int height);

protected:
	void InitMonitor();

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
	LazyDetector* lazyDetector;
	DetectResultSegment detectResult;
	DetectResultWithTrackerStatus detectResultWithStatus;

	FourLimits* allCandidateTargets; // this is an reference of all detected targets, the space of these target candidates is init and release in detetor, DON'T init or release this pointer !!!
	int allCandidateTargetsCount;

	Tracker* TrackerList;
};

inline Monitor::Monitor(const int width, const int height, const int dilationRadius, const int discretizationScale)
	: currentFrame(nullptr),
	  Width(width),
	  Height(height),
	  DilationRadius(dilationRadius),
	  DiscretizationScale(discretizationScale),
	  confidences(nullptr),
	  detector(nullptr),
	  lazyDetector(nullptr),
	  allCandidateTargets(nullptr),
	  allCandidateTargetsCount(0)
{
	// 初始化Monitor
	InitMonitor();

	// 初始化置信值
	InitConfidenceValueMap();

	// 初始化跟踪器列表
	InitTrackerList();
}

inline Monitor::~Monitor()
{
	delete detector;
	delete lazyDetector;
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

inline void Monitor::UpdateTracker(Tracker& tracker, const TargetPosition& targetPos, const TargetInfo& targetInfo, bool isExtendLifetime)
{
	tracker.Postion.bottomRightY = targetPos.bottomRightY;
	tracker.Postion.bottomRightX = targetPos.bottomRightX;
	tracker.Postion.topLeftX = targetPos.topLeftX;
	tracker.Postion.topLeftY = targetPos.topLeftY;

	tracker.Info = targetInfo;

	int bc = 0;
	int br = 0;
	GetBlockPos(targetPos, br, bc);
	tracker.BlockX = bc;
	tracker.BlockY = br;

	Point centerPos((targetPos.topLeftX + targetPos.bottomRightX) / 2, (targetPos.topLeftY + targetPos.bottomRightY) / 2);
	if(isExtendLifetime == true)
		tracker.ExtendLifeTime(centerPos);
}

inline void Monitor::AddTracker(const TargetPosition& targetPos, const TargetInfo& targetInfo)
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

			Point centerPos((targetPos.topLeftX + targetPos.bottomRightX) / 2, (targetPos.topLeftY + targetPos.bottomRightY) / 2);

			TrackerList[i].Info = targetInfo;

			TrackerList[i].InitLifeTime(centerPos);
			TrackerList[i].ValidFlag = true;
			break;
		}
	}
}

inline void Monitor::UpdateTrackerForAllBlocks(unsigned short* frame)
{
	// Step 1. check if has tracker

	// Step 2. if has tracker, then update all tracker for their target

	// Step 3. if there no tracker, or there are no target has been tracked, then add tracker for them

	// This is step 1;
	auto hasTracker = false;
	for (auto i = 0; i < MaxTrackerCount; ++i)
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
		for (auto i = 0; i < MaxTrackerCount; ++i)
		{
			// 只更新有效的跟踪器
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

				auto updateTrackerInfoSuccess = false;
				auto maxAreaDiff = 65535;
				auto mostLikelyTargetIndex = 0;
				for (auto j = 0; j < detectResultWithStatus.detectResultPointers->targetCount; ++j)
				{
					if (detectResultWithStatus.hasTracker[j] == true)
						continue;
					auto targetCenterX = (detectResultWithStatus.detectResultPointers->targets[j].bottomRightX + detectResultWithStatus.detectResultPointers->targets[j].topLeftX) / 2;
					auto targetCenterY = (detectResultWithStatus.detectResultPointers->targets[j].bottomRightY + detectResultWithStatus.detectResultPointers->targets[j].topLeftY) / 2;
					// if find one target in this search region, then update this tracker info, actually need find all target int this region, then the nearest target should be last result
					// update: we use area nearly equal method
					if (targetCenterX > searchRegionLeft && targetCenterX < searchRegionRight && targetCenterY > searchRegionTop && targetCenterY < searchRegionBottom)
					{
						if (maxAreaDiff > std::abs(detectResultWithStatus.detectResultPointers->targetInfo[j].area - TrackerList[i].Info.area))
						{
							maxAreaDiff = std::abs(detectResultWithStatus.detectResultPointers->targetInfo[j].area - TrackerList[i].Info.area);
							mostLikelyTargetIndex = j;
						}
						updateTrackerInfoSuccess = true;
					}
				}
				if (updateTrackerInfoSuccess == true)
				{
					UpdateTracker(TrackerList[i],
						detectResultWithStatus.detectResultPointers->targets[mostLikelyTargetIndex],
						detectResultWithStatus.detectResultPointers->targetInfo[mostLikelyTargetIndex]);
					detectResultWithStatus.hasTracker[mostLikelyTargetIndex] = true;
				}
				// if tracker cannot find one target in it's search region, then shrink it's lifetime
				// update: if there no target detect in this region, need tracker research target
				if (updateTrackerInfoSuccess == false)
				{
					auto targetWidth = TrackerList[i].Postion.bottomRightX - TrackerList[i].Postion.topLeftX + 1;
					auto targetHeight = TrackerList[i].Postion.bottomRightY - TrackerList[i].Postion.topLeftY + 1;

					auto maxContrast = 0.0;
					int maxR = searchRegionTop;
					int maxC = searchRegionLeft;
					TargetInfo info;
					for (auto r = searchRegionTop; r <= searchRegionBottom - targetHeight; ++r)
					{
						for (auto c = searchRegionLeft; c <= searchRegionRight - targetWidth; ++c)
						{
							auto currentContrast = GetContrastRate(frame, c, r, targetWidth, targetHeight);
							auto Area = GetArea(frame, c, r, targetWidth, targetHeight);
							if (currentContrast > maxContrast)
							{
								maxContrast = currentContrast;
								maxR = r;
								maxC = c;
								info.area = Area;
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
						pos.bottomRightX = maxC + targetWidth - 1;
						pos.bottomRightY = maxR + targetHeight - 1;
						UpdateTracker(TrackerList[i], pos, info, false);
					}
				}
			}
		}
	}
	// This is step 3 : Check all detect targets
	for (auto i = 0; i < detectResultWithStatus.detectResultPointers->targetCount; ++i)
	{
		// if this detected target already has tracker for it, do nothing
		if (detectResultWithStatus.hasTracker[i] == true)
			continue;

		// if have no tracker for this target, calculate it's block postion
		int br = 0;
		int bc = 0;
		GetBlockPos(detectResultWithStatus.detectResultPointers->targets[i], br, bc);

		auto TotalConfValue = this->ConfidenceValueMap[br * BlockCols + bc] + CalculateQueueSum(bc, br);
		if (TotalConfValue >= TrackConfirmThreshold)
		{
			AddTracker(detectResultWithStatus.detectResultPointers->targets[i], detectResultWithStatus.detectResultPointers->targetInfo[i]);
			detectResultWithStatus.hasTracker[i] = true;
		}
	}
}

inline double Monitor::GetContrastRate(unsigned short* frame, int left, int top, int width, int height)
{
	auto widthPadding = width;
	auto heightPadding = height;

	auto avgSurrouding = 0.0;
	auto maxTarget = 0.0;

	// target max value
	for (auto r = top; r < top + height; ++r)
	{
		for (auto c = left; c < left + width; ++c)
		{
			if (maxTarget < static_cast<double>(frame[r * Width + c]))
				maxTarget = static_cast<double>(frame[r * Width + c]);
		}
	}

	// target average gray value
	auto sum = 0.0;
	for (auto r = top; r < top + height; ++r)
	{
		auto sumRow = 0.0;
		for (auto c = left; c < left + width; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += (sumRow / width);
	}
	auto avgTarget = sum / height;

	// target surrounding average gray value
	sum = 0.0;
	auto surroundingTop = top - heightPadding;
	surroundingTop = surroundingTop < 0 ? 0 : surroundingTop;
	auto surroundingLeft = left - widthPadding;
	surroundingLeft = surroundingLeft < 0 ? 0 : surroundingLeft;
	auto surroundingRight = left + width + widthPadding;
	surroundingRight = surroundingRight > Width ? Width : surroundingRight;
	auto surroundingBottom = top + height + heightPadding;
	surroundingBottom = surroundingBottom > Height ? Height : surroundingBottom;
	for (auto r = surroundingTop; r < top; ++r)
	{
		auto sumRow = 0.0;
		for (auto c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (auto r = top + height; r < surroundingBottom; ++r)
	{
		auto sumRow = 0.0;
		for (auto c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (auto r = top; r < top + height; ++r)
	{
		auto sumRow = 0.0;
		for (auto c = surroundingLeft; c < left; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		for (auto c = left + width; c < surroundingRight; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += sumRow / ((left - surroundingLeft) + (surroundingRight - (left + width)));
	}
	avgSurrouding = sum / (surroundingBottom - surroundingTop);

	auto result = maxTarget - avgSurrouding;
	return result;
}

inline int Monitor::GetArea(unsigned short* frame, int left, int top, int width, int height)
{
	auto avgTarget = 0.0;
	auto area = 0;

	// target average gray value
	auto sum = 0.0;
	for (auto r = top; r < top + height; ++r)
	{
		auto sumRow = 0.0;
		for (auto c = left; c < left + width; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += (sumRow / width);
	}
	avgTarget = sum / height;

	for (auto r = top; r < top + height; ++r)
	{
		for (auto c = left; c < left + width; ++c)
		{
			if (static_cast<double>(frame[r * Width + c]) > avgTarget)
				area++;
		}
	}

	return area;
}

/******************************************************************************
/*  Main Process Method
******************************************************************************/
inline bool Monitor::Process(unsigned short* frame, DetectResultSegment* result)
{
	// detect target in single frame
	detector->DetectTargets(frame, &detectResult, &this->allCandidateTargets, &this->allCandidateTargetsCount);
	// lazyDetector->DetectTargets(frame, &detectResult);

	// copy detect result and set default tracking status
	detectResultWithStatus.detectResultPointers = &detectResult;
	memset(detectResultWithStatus.hasTracker, false, sizeof(bool) * MAX_DETECTED_TARGET_COUNT);

	// reset current frame block mask map
	ResetCurrentDetectMask();

	// update confidence value map and confidence queue map
	UpdateConfidenceValueAndUpdateConfidenceQueue();

	// update tracker for all blocks
	UpdateTrackerForAllBlocks(frame);

	// copy tracking result back
	memcpy(result->header, detectResult.header, FRAME_HEADER_LENGTH);
	result->targetCount = detectResult.targetCount;

	std::sort(TrackerList, TrackerList + MaxTrackerCount, CompareTracker);

	auto trackingTargetCount = 0;
	for (auto i = 0; i < MaxTrackerCount; ++i)
	{
		if (TrackerList[i].ValidFlag == true && TrackerList[i].LifeTime > 2)
		{
			// if (TrackerList[i].IsNotMove())
			// 	continue;
			if (TrackerList[i].IsComming() == false)
			{
				result->targets[trackingTargetCount] = TrackerList[i].Postion;
				trackingTargetCount++;
				if (trackingTargetCount >= MAX_DETECTED_TARGET_COUNT)
				{
					break;
				}
			}
		}
	}
	result->targetCount = trackingTargetCount;
	return true;
}

inline void Monitor::InitMonitor()
{
	// 计算块的行数和列数
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

	this->lazyDetector = new LazyDetector(Width, Height, DilationRadius, DiscretizationScale);
	this->lazyDetector->InitDetector();
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
