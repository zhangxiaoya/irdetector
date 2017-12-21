#ifndef __MULTI_SEARCHER_H__
#define __MULTI_SEARCHER_H__

#include "../Models/CandidateTargets.hpp"
#include "../LogPrinter/LogPrinter.hpp"
#include "../Detector/Detector.hpp"
#include "../Checkers/CheckPerf.h"
#include "../Headers/SearcherParameters.h"
#include "Monitor.hpp"

/********************************************************************************/
/* 旋转搜索类定义                                                                */
/********************************************************************************/
class MultiSearcher
{
public:
	MultiSearcher(const int width,
		const int height,
		const int pixelSize,
		const int dilationRadius,
		const int discretizationScale);

	~MultiSearcher();

	/********************************************************************************/
	/* 搜索一圈函数声明（外部接口）                                                   */
	/********************************************************************************/
	void SearchOneRound(unsigned short* frameData, DetectResultSegment* result);

	/********************************************************************************/
	/* 初始化资源函数声明                                                            */
	/********************************************************************************/
	void Init();

private:
	/********************************************************************************/
	/* 释放资源函数声明                                                              */
	/********************************************************************************/
	void Release();

	/********************************************************************************/
	/* 检查是否是地物背景区域函数声明                                                  */
	/********************************************************************************/
	bool CheckIfHaveGroundObject(int frame_index) const;

	/********************************************************************************/
	/* 获取一圈检测最可能是目标函数声明                                                */
	/********************************************************************************/
	void GetLastResultAfterOneRound();

protected:
	/********************************************************************************/
	/* 计算目标信杂比函数声明                                                         */
	/********************************************************************************/
	double CalculateSCR(unsigned short* frame, const TargetPosition& target);

	/********************************************************************************/
	/* 计算候选目标局部差值函数声明                                                 */
	/********************************************************************************/
	double CalculateLocalDiff(unsigned short* frame, const TargetPosition& target);

	/********************************************************************************/
	/* 计算检测目标最大值函数声明                                                     */
	/********************************************************************************/
	double CalculateMaxValue(unsigned short* frame, const TargetPosition& target) const;

	/********************************************************************************/
	/* 计算目标分值，并存储到候选队列函数声明                                          */
	/********************************************************************************/
	void CalculateScoreForDetectedTargetsAndPushToCandidateQueue(unsigned short* frame, const DetectResultSegment& result, int frameIndex);

	/********************************************************************************/
	/* 比较函数定义                                                                  */
	/********************************************************************************/
	static bool CompareCandidates(CandidateTarget& a, CandidateTarget& b)
	{
		return  a.score > b.score;
	}

private:
	// 目标检测器指针声明
	Detector* detector;

	// 目标跟踪器
	Monitor* monitors[FRAME_COUNT_ONE_ROUND];

	// 一圈内所有图像
	unsigned short* FramesInOneRound[FRAME_COUNT_ONE_ROUND];

	// 图像参数
	int Width;
	int Height;
	int PixelSize;
	// 预处理参数
	int DilationRadius;
	int DiscretizationScale;

	// 一圈检测目标存储队列
	CandidateTarget AllCandidateTargets[SEARCH_TARGET_COUNT_ONE_ROUND];
	// 存储队列中目标个数
	int CandidateTargetCount;
	// 帧数指示器，每一圈重置
	int FrameIndex;

	// 单帧检测临时存储结果
	DetectResultSegment resultOfSingleFrame[SEARCH_TARGET_COUNT_ONE_ROUND];
};

inline MultiSearcher::MultiSearcher(const int width,
	const int height,
	const int pixelSize,
	const int dilationRadius,
	const int discretizationScale) :
	detector(nullptr),
	Width(width),
	Height(height),
	PixelSize(pixelSize),
	DilationRadius(dilationRadius),
	DiscretizationScale(discretizationScale),
	CandidateTargetCount(0),
	FrameIndex(0)
{
	// Init();
}

inline MultiSearcher::~MultiSearcher()
{
	Release();
}

/********************************************************************************/
/* 初始化资源函数定义                                                             */
/********************************************************************************/
inline void MultiSearcher::Init()
{
	// 初始化检测器
	this->detector = new Detector(Width, Height, DilationRadius, DiscretizationScale);
	detector->InitSpace();
	detector->SetRemoveFalseAlarmParameters(true, false, false, false, true, true);

	// 初始化每个角度对应的跟踪器
	for (int i = 0; i < FRAME_COUNT_ONE_ROUND; ++i)
	{
		this->monitors[i] = new Monitor(Width, Height, DilationRadius, DiscretizationScale);
		this->monitors[i]->InitDetector();
	}

	// 初始化存储一圈帧图像空间
	for (auto i = 0; i < FRAME_COUNT_ONE_ROUND; ++i)
	{
		FramesInOneRound[i] = new unsigned short[Width * Height];
	}
}

/********************************************************************************/
/* 释放资源函数定义                                                              */
/********************************************************************************/
inline void MultiSearcher::Release()
{
	// 释放存储一圈图像空间
	for (auto i = 0; i < FRAME_COUNT_ONE_ROUND; ++i)
	{
		delete FramesInOneRound[i];
	}
	// 删除检测器
	delete detector;
	// 删除每个角度的跟踪器
	for (int i = 0; i < FRAME_COUNT_ONE_ROUND; ++i)
	{
		delete this->monitors[i];
	}
}

/********************************************************************************/
/* 检测是否是地物背景函数定义                                                     */
/********************************************************************************/
inline bool MultiSearcher::CheckIfHaveGroundObject(int frame_index) const
{
	// 测试文件使用
	if (frame_index >= 64 && frame_index < 145)
		return true;
	return false;
}

/********************************************************************************/
/* 获取最终一圈后结果函数定义                                                     */
/********************************************************************************/
inline void MultiSearcher::GetLastResultAfterOneRound()
{
	// 候选队列中的所有检测到的目标，按照分值排序
	std::sort(this->AllCandidateTargets, AllCandidateTargets + CandidateTargetCount, CompareCandidates);

	// 初始化帧显示标记（测试用）
	int flag[FRAME_COUNT_WHICH_MOST_LIKELY_HAVE_TARGETS] = { -1 };
	// 初始化结果帧图像 （测试用）
	cv::Mat imgs[FRAME_COUNT_WHICH_MOST_LIKELY_HAVE_TARGETS];

	// 最终一圈检测到的目标个数
	int targetCount = 0;
	// 帧数计数器
	int frameCount = 0;

	// 检测预定帧数，每一帧可能包含多个结果
	while (frameCount < FRAME_COUNT_WHICH_MOST_LIKELY_HAVE_TARGETS)
	{
		auto existFlag = false;
		for (auto i = 0; i < frameCount; ++i)
		{
			if (flag[i] == AllCandidateTargets[targetCount].frameIndex)
			{
				existFlag = true;
				break;
			}
		}
		if (existFlag == true)
		{
			targetCount++;
			continue;
		}

		flag[frameCount] = AllCandidateTargets[targetCount].frameIndex;
		frameCount++;
		targetCount++;
	}

	// 再从剩下的检测结果中，选择在当前的几帧图像中，但是分值比较低的，也有可能是目标

	// 临时画出结果（调试用）
	char windowsName[100];
	std::string windowsNameFormat = "Last Result Frame %d";

	for (auto i = 0; i< frameCount; ++i)
	{
		auto initImage = false;
		for (auto j = 0; j< targetCount; ++j)
		{
			if (AllCandidateTargets[j].frameIndex == flag[i] && initImage == false)
			{
				ShowFrame::ToMat<unsigned short>(FramesInOneRound[AllCandidateTargets[j].frameIndex], Width, Height, imgs[i], CV_8UC3);
				initImage = true;
			}
			if (AllCandidateTargets[j].frameIndex == flag[i])
				rectangle(imgs[i], cv::Point(AllCandidateTargets[j].left - 5, AllCandidateTargets[j].top - 5), cv::Point(AllCandidateTargets[j].right + 5, AllCandidateTargets[j].bottom + 5), cv::Scalar(255, 255, 0));
		}
		std::cout << "index is " << std::setw(4) << flag[i] << std::endl;

		sprintf(windowsName, windowsNameFormat.c_str(), i);
		imshow(windowsName, imgs[i]);

		cv::waitKey(1000);
	}
}

/********************************************************************************/
/* 外部接口函数定义                                                              */
/********************************************************************************/
inline void MultiSearcher::SearchOneRound(unsigned short* frameData, DetectResultSegment* result)
{
	memcpy(FramesInOneRound[FrameIndex], frameData, Width * Height * PixelSize);

	monitors[FrameIndex]->Process(FramesInOneRound[FrameIndex], &resultOfSingleFrame[FrameIndex]);

	//CalculateScoreForDetectedTargetsAndPushToCandidateQueue(FramesInOneRound[FrameIndex], resultOfSingleFrame[FrameIndex], FrameIndex);
	memcpy(result, &resultOfSingleFrame[FrameIndex], sizeof(DetectResultSegment));

	FrameIndex++;

	if (FrameIndex == FRAME_COUNT_ONE_ROUND)
	{
		// GetLastResultAfterOneRound();
		// CandidateTargetCount = 0;
		FrameIndex = 0;
	}

}

inline double MultiSearcher::CalculateSCR(unsigned short* frame, const TargetPosition& target)
{
	int width = target.bottomRightX - target.topLeftX;
	int height = target.bottomRightY - target.topLeftY;

	int widthPadding = width;
	int heightPadding = height;

	double avgTarget = 0.0;
	double avgSurrouding = 0.0;

	// target average gray value
	double sum = 0.0;
	for (int r = target.topLeftY; r < target.bottomRightY; ++r)
	{
		double sumRow = 0.0;
		for (int c = target.topLeftX; c < target.bottomRightX; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += (sumRow / width);
	}
	avgTarget = sum / height;

	// target surrounding average gray value
	sum = 0.0;
	int surroundingTop = target.topLeftY - heightPadding;
	surroundingTop = surroundingTop < 0 ? 0 : surroundingTop;
	int surroundingLeft = target.topLeftX - widthPadding;
	surroundingLeft = surroundingLeft < 0 ? 0 : surroundingLeft;
	int surroundingRight = target.bottomRightX + widthPadding;
	surroundingRight = surroundingRight > Width ? Width : surroundingRight;
	int surroundingBottom = target.bottomRightY + heightPadding;
	surroundingBottom = surroundingBottom > Height ? Height : surroundingBottom;
	for (int r = surroundingTop; r < target.topLeftY; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (int r = target.bottomRightY; r < surroundingBottom; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (int r = target.topLeftY; r < target.bottomRightY; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < target.topLeftX; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		for (int c = target.bottomRightX; c < surroundingRight; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += sumRow / ((target.topLeftX - surroundingLeft) + (surroundingRight - target.bottomRightX));
	}
	avgSurrouding = sum / (surroundingBottom - surroundingTop);

	// calculate standard deviation
	sum = 0.0;
	for (int r = surroundingTop; r < target.topLeftY; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += (static_cast<double>(frame[r * Width + c]) - avgSurrouding) * (static_cast<double>(frame[r * Width + c]) - avgSurrouding);
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (int r = target.bottomRightY; r < surroundingBottom; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += (static_cast<double>(frame[r * Width + c]) - avgSurrouding) * (static_cast<double>(frame[r * Width + c]) - avgSurrouding);
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (int r = target.topLeftY; r < target.bottomRightY; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < target.topLeftX; ++c)
		{
			sumRow += (static_cast<double>(frame[r * Width + c]) - avgSurrouding) * (static_cast<double>(frame[r * Width + c]) - avgSurrouding);
		}
		for (int c = target.bottomRightX; c < surroundingRight; ++c)
		{
			sumRow += (static_cast<double>(frame[r * Width + c]) - avgSurrouding) * (static_cast<double>(frame[r * Width + c]) - avgSurrouding);
		}
		sum += sumRow / ((target.topLeftX - surroundingLeft) + (surroundingRight - target.bottomRightX));
	}

	double stdDivation = sqrt(sum / (surroundingBottom - surroundingTop));

	return abs(avgTarget - avgSurrouding) / stdDivation;
}

inline double MultiSearcher::CalculateLocalDiff(unsigned short* frame, const TargetPosition& target)
{
	int objectWidth = target.bottomRightX - target.topLeftX;
	int objectHeight = target.bottomRightY - target.topLeftY;

	int widthPadding = objectWidth;
	int heightPadding = objectHeight;

	double maxTarget = CalculateMaxValue(frame, target);
	double avgSurrouding = 0.0;

	// target surrounding average gray value
	double sum = 0.0;
	int surroundingTop = target.topLeftY - heightPadding;
	surroundingTop = surroundingTop < 0 ? 0 : surroundingTop;
	int surroundingLeft = target.topLeftX - widthPadding;
	surroundingLeft = surroundingLeft < 0 ? 0 : surroundingLeft;
	int surroundingRight = target.bottomRightX + widthPadding;
	surroundingRight = surroundingRight > Width ? Width : surroundingRight;
	int surroundingBottom = target.bottomRightY + heightPadding;
	surroundingBottom = surroundingBottom > Height ? Height : surroundingBottom;

	for (int r = surroundingTop; r < target.topLeftY; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (int r = target.bottomRightY; r < surroundingBottom; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < surroundingRight; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += sumRow / (surroundingRight - surroundingLeft);
	}
	for (int r = target.topLeftY; r < target.bottomRightY; ++r)
	{
		double sumRow = 0.0;
		for (int c = surroundingLeft; c < target.topLeftX; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		for (int c = target.bottomRightX; c < surroundingRight; ++c)
		{
			sumRow += static_cast<double>(frame[r * Width + c]);
		}
		sum += sumRow / ((target.topLeftX - surroundingLeft) + (surroundingRight - target.bottomRightX));
	}
	avgSurrouding = sum / (surroundingBottom - surroundingTop);

	return maxTarget - avgSurrouding;
}

inline double MultiSearcher::CalculateMaxValue(unsigned short* frame, const TargetPosition& target) const
{
	double max = 0.0;
	for (auto r = target.topLeftY; r < target.bottomRightY; ++r)
	{
		for (auto c = target.topLeftX; c < target.bottomRightX; ++c)
		{
			if (max < frame[r * Width + c])
				max = static_cast<double>(frame[r * Width + c]);
		}
	}
	return max;
}

inline void MultiSearcher::CalculateScoreForDetectedTargetsAndPushToCandidateQueue(unsigned short* frame, const DetectResultSegment& result, int frameIndex)
{
	for (auto i = 0; i< result.targetCount; i++)
	{
		AllCandidateTargets[CandidateTargetCount].left = result.targets[i].topLeftX;
		AllCandidateTargets[CandidateTargetCount].right = result.targets[i].bottomRightX;
		AllCandidateTargets[CandidateTargetCount].top = result.targets[i].topLeftY;
		AllCandidateTargets[CandidateTargetCount].bottom = result.targets[i].bottomRightY;
		AllCandidateTargets[CandidateTargetCount].frameIndex = frameIndex;
		// AllCandidateTargets[CandidateTargetCount].score = CalculateMaxValue(frame, result.targets[i]);
		AllCandidateTargets[CandidateTargetCount].score = CalculateLocalDiff(frame, result.targets[i]);
		CandidateTargetCount++;
	}
}

#endif
