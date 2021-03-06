#pragma once

#ifndef DecisionTime
#define DecisionTime (3)
#endif // !DecisionTime

#ifndef LazyCountorCount
#define LazyCountorCount (DecisionTime * MAX_DETECTED_TARGET_COUNT)
#endif // !LazyQueueCount

#ifndef MaxCheckerLifeTime
#define MaxCheckerLifeTime (4)
#endif // !MaxCheckerLifeTime

#include "Detector.hpp"
#include "../Models/DetectedTargetLazyChecker.hpp"
/*****************************************************************************************************************************/
/* 延迟搜索类定义                                                                                                            */
/*****************************************************************************************************************************/
class LazyDetector
{
public:
	LazyDetector(const int width, const int height, const int dilationRadius, const int discretizationScale);

	~LazyDetector();

	void InitDetector();

	void DetectTargets(unsigned short* frame, DetectResultSegment* result);

	void ResetForbiddenZone();

	FourLimits* GetCurrentForbiddenZones(int& forbiddenZoneCount);

private:
	/*检测器声明*/
	Detector* detector;

	/*检测器输入参数信息声明*/
	int Width;
	int Height;
	int DialationRadius;
	int DiscretizationScale;

	/*检测目标计数器*/
	DetectedTargetLazyChecker lazyChecker[LazyCountorCount];
};

inline LazyDetector::LazyDetector(const int width, const int height, const int dilationRadius, const int discretizationScale)
	:detector(nullptr),
	 Width(width),
	 Height(height),
	 DialationRadius(dilationRadius),
	 DiscretizationScale(discretizationScale)
{
}

inline LazyDetector::~LazyDetector()
{
	delete detector;
}

inline void LazyDetector::ResetForbiddenZone()
{
	this->detector->ResetForbiddenZones();
}

inline FourLimits* LazyDetector::GetCurrentForbiddenZones(int& forbiddenZoneCount)
{
	return this->detector->GetCurrentForbiddenZones(forbiddenZoneCount);
}

inline void LazyDetector::InitDetector()
{
	this->detector = new Detector(Width, Height, DialationRadius, DiscretizationScale);
	detector->InitSpace();
	detector->SetRemoveFalseAlarmParameters(false, false, false, false, true, true);
}

inline void LazyDetector::DetectTargets(unsigned short* frame, DetectResultSegment* result)
{
	this->detector->DetectTargets(frame, result);

	for (auto detectTargetIdx = 0; detectTargetIdx < result->targetCount; ++detectTargetIdx)
	{
		if (result->targets[detectTargetIdx].bottomRightX - result->targets[detectTargetIdx].topLeftX > 3 &&
		result->targets[detectTargetIdx].bottomRightY - result->targets[detectTargetIdx].topLeftY > 3)
			continue;

		auto existFlag = false;
		for (auto countorIdx = 0; countorIdx < LazyCountorCount; ++countorIdx)
		{
			if (lazyChecker[countorIdx].lifeTime == 0)
				continue;

			if (Util::IsSameTarget(FourLimits(result->targets[detectTargetIdx]), lazyChecker[countorIdx].position))
			{
				lazyChecker[countorIdx].count++;
				existFlag = true;
				break;
			}
		}
		if (existFlag == false)
		{
			for (auto countorIdx = 0; countorIdx < LazyCountorCount; ++countorIdx)
			{
				if (lazyChecker[countorIdx].lifeTime != 0)
					continue;

				lazyChecker[countorIdx].position = result->targets[detectTargetIdx];
				lazyChecker[countorIdx].lifeTime = 1;
				lazyChecker[countorIdx].count = 1;
				break;
			}
		}
	}

	// 累积
	for (auto i = 0; i < LazyCountorCount; ++i)
	{
		if (lazyChecker[i].count >= 3)
		{
			// add forbidden zone
			if(detector->AddForbiddenZone(lazyChecker[i].position) == false)
				printf("Add Forbidden Failed, May be the bad point is too many!");
			lazyChecker[i].count = lazyChecker[i].lifeTime = 0;
			lazyChecker[i].position = FourLimits();
			continue;
		}
		if (lazyChecker[i].count > 0)
			lazyChecker[i].lifeTime++;
		if (lazyChecker[i].lifeTime == 4)
		{
			lazyChecker[i].count = lazyChecker[i].lifeTime = 0;
			lazyChecker[i].position = FourLimits();
		}
	}
}
