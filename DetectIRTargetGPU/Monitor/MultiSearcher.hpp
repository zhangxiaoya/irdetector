#ifndef __MULTI_SEARCHER_H__
#define __MULTI_SEARCHER_H__

#include "../Models/CandidateTargets.hpp"
#include "../LogPrinter/LogPrinter.hpp"
#include "../Detector/Detector.hpp"
#include "../Checkers/CheckPerf.h"
#include "../Headers/SearcherParameters.h"
#include "Monitor.hpp"

/********************************************************************************/
/* ��ת�����ඨ��                                                                */
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
	/* ����һȦ�����������ⲿ�ӿڣ�                                                   */
	/********************************************************************************/
	void SearchOneRound(unsigned short* frameData, DetectResultSegment* result);

	/********************************************************************************/
	/* ��ʼ����Դ��������                                                            */
	/********************************************************************************/
	void Init();

private:
	/********************************************************************************/
	/* �ͷ���Դ��������                                                              */
	/********************************************************************************/
	void Release();

	/********************************************************************************/
	/* ����Ƿ��ǵ��ﱳ������������                                                  */
	/********************************************************************************/
	bool CheckIfHaveGroundObject(int frame_index) const;

	/********************************************************************************/
	/* ��ȡһȦ����������Ŀ�꺯������                                                */
	/********************************************************************************/
	void GetLastResultAfterOneRound();

protected:
	/********************************************************************************/
	/* ����Ŀ�����ӱȺ�������                                                         */
	/********************************************************************************/
	double CalculateSCR(unsigned short* frame, const TargetPosition& target);

	/********************************************************************************/
	/* �����ѡĿ��ֲ���ֵ��������                                                 */
	/********************************************************************************/
	double CalculateLocalDiff(unsigned short* frame, const TargetPosition& target);

	/********************************************************************************/
	/* ������Ŀ�����ֵ��������                                                     */
	/********************************************************************************/
	double CalculateMaxValue(unsigned short* frame, const TargetPosition& target) const;

	/********************************************************************************/
	/* ����Ŀ���ֵ�����洢����ѡ���к�������                                          */
	/********************************************************************************/
	void CalculateScoreForDetectedTargetsAndPushToCandidateQueue(unsigned short* frame, const DetectResultSegment& result, int frameIndex);

	/********************************************************************************/
	/* �ȽϺ�������                                                                  */
	/********************************************************************************/
	static bool CompareCandidates(CandidateTarget& a, CandidateTarget& b)
	{
		return  a.score > b.score;
	}

private:
	// Ŀ������ָ������
	Detector* detector;

	// Ŀ�������
	Monitor* monitors[FRAME_COUNT_ONE_ROUND];

	// һȦ������ͼ��
	unsigned short* FramesInOneRound[FRAME_COUNT_ONE_ROUND];

	// ͼ�����
	int Width;
	int Height;
	int PixelSize;
	// Ԥ�������
	int DilationRadius;
	int DiscretizationScale;

	// һȦ���Ŀ��洢����
	CandidateTarget AllCandidateTargets[SEARCH_TARGET_COUNT_ONE_ROUND];
	// �洢������Ŀ�����
	int CandidateTargetCount;
	// ֡��ָʾ����ÿһȦ����
	int FrameIndex;

	// ��֡�����ʱ�洢���
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
/* ��ʼ����Դ��������                                                             */
/********************************************************************************/
inline void MultiSearcher::Init()
{
	// ��ʼ�������
	this->detector = new Detector(Width, Height, DilationRadius, DiscretizationScale);
	detector->InitSpace();
	detector->SetRemoveFalseAlarmParameters(true, false, false, false, true, true);

	// ��ʼ��ÿ���Ƕȶ�Ӧ�ĸ�����
	for (int i = 0; i < FRAME_COUNT_ONE_ROUND; ++i)
	{
		this->monitors[i] = new Monitor(Width, Height, DilationRadius, DiscretizationScale);
		this->monitors[i]->InitDetector();
	}

	// ��ʼ���洢һȦ֡ͼ��ռ�
	for (auto i = 0; i < FRAME_COUNT_ONE_ROUND; ++i)
	{
		FramesInOneRound[i] = new unsigned short[Width * Height];
	}
}

/********************************************************************************/
/* �ͷ���Դ��������                                                              */
/********************************************************************************/
inline void MultiSearcher::Release()
{
	// �ͷŴ洢һȦͼ��ռ�
	for (auto i = 0; i < FRAME_COUNT_ONE_ROUND; ++i)
	{
		delete FramesInOneRound[i];
	}
	// ɾ�������
	delete detector;
	// ɾ��ÿ���Ƕȵĸ�����
	for (int i = 0; i < FRAME_COUNT_ONE_ROUND; ++i)
	{
		delete this->monitors[i];
	}
}

/********************************************************************************/
/* ����Ƿ��ǵ��ﱳ����������                                                     */
/********************************************************************************/
inline bool MultiSearcher::CheckIfHaveGroundObject(int frame_index) const
{
	// �����ļ�ʹ��
	if (frame_index >= 64 && frame_index < 145)
		return true;
	return false;
}

/********************************************************************************/
/* ��ȡ����һȦ������������                                                     */
/********************************************************************************/
inline void MultiSearcher::GetLastResultAfterOneRound()
{
	// ��ѡ�����е����м�⵽��Ŀ�꣬���շ�ֵ����
	std::sort(this->AllCandidateTargets, AllCandidateTargets + CandidateTargetCount, CompareCandidates);

	// ��ʼ��֡��ʾ��ǣ������ã�
	int flag[FRAME_COUNT_WHICH_MOST_LIKELY_HAVE_TARGETS] = { -1 };
	// ��ʼ�����֡ͼ�� �������ã�
	cv::Mat imgs[FRAME_COUNT_WHICH_MOST_LIKELY_HAVE_TARGETS];

	// ����һȦ��⵽��Ŀ�����
	int targetCount = 0;
	// ֡��������
	int frameCount = 0;

	// ���Ԥ��֡����ÿһ֡���ܰ���������
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

	// �ٴ�ʣ�µļ�����У�ѡ���ڵ�ǰ�ļ�֡ͼ���У����Ƿ�ֵ�Ƚϵ͵ģ�Ҳ�п�����Ŀ��

	// ��ʱ��������������ã�
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
/* �ⲿ�ӿں�������                                                              */
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
