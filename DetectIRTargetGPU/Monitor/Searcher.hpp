#ifndef __SEARCHER_H__
#define __SEARCHER_H__

#include "../Models/CandidateTargets.hpp"
#include "../LogPrinter/LogPrinter.hpp"
#include "../Detector/Detector.hpp"
#include "../Checkers/CheckPerf.h"
#include "../Headers/SearcherParameters.h"

/********************************************************************************/
/* ��ת�����ඨ��                                                                */
/********************************************************************************/
class Searcher
{
public:
	Searcher(const int width,
		const int height,
		const int pixelSize,
		const int dilationRadius,
		const int discretizationScale);

	~Searcher();

	/********************************************************************************/
	/* ����һȦ�����������ⲿ�ӿڣ�                                                   */
	/********************************************************************************/
	void SearchOneRound(unsigned short* frameData);

private:
	/********************************************************************************/
	/* ��ʼ����Դ��������                                                            */
	/********************************************************************************/
	void Init();

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
	int candidateTargetCount;
	// ֡��ָʾ����ÿһȦ����
	int frameIndex;

	// ��֡�����ʱ�洢���
	DetectResultSegment resultOfSingleFrame;
};

inline Searcher::Searcher(const int width,
                          const int height,
                          const int pixelSize,
                          const int dilationRadius,
                          const int discretizationScale):
	detector(nullptr),
	Width(width),
	Height(height),
	PixelSize(pixelSize),
	DilationRadius(dilationRadius),
	DiscretizationScale(discretizationScale),
	candidateTargetCount(0),
	frameIndex(0)
{
	Init();
}

inline Searcher::~Searcher()
{
	Release();
}

/********************************************************************************/
/* ��ʼ����Դ��������                                                             */
/********************************************************************************/
inline void Searcher::Init()
{
	// ��ʼ�������
	this->detector = new Detector(Width, Height, DilationRadius, DiscretizationScale);
	detector->InitSpace();
	detector->SetRemoveFalseAlarmParameters(true, false, false, false, true, true);

	// ��ʼ���洢һȦ֡ͼ��ռ�
	for (auto i = 0; i < FRAME_COUNT_ONE_ROUND; ++i)
	{
		FramesInOneRound[i] = new unsigned short[Width * Height];
	}
}

/********************************************************************************/
/* �ͷ���Դ��������                                                              */
/********************************************************************************/
inline void Searcher::Release()
{
	// �ͷŴ洢һȦͼ��ռ�
	for (auto i = 0; i < FRAME_COUNT_ONE_ROUND; ++i)
	{
		delete FramesInOneRound[i];
	}
	// ɾ�������
	delete detector;
}

/********************************************************************************/
/* ����Ƿ��ǵ��ﱳ����������                                                     */
/********************************************************************************/
inline bool Searcher::CheckIfHaveGroundObject(int frame_index) const
{
	// �����ļ�ʹ��
	if (frame_index >= 64 && frame_index < 145)
		return true;
	return false;
}

/********************************************************************************/
/* ��ȡ����һȦ������������                                                     */
/********************************************************************************/
inline void Searcher::GetLastResultAfterOneRound()
{
	// ��ѡ�����е����м�⵽��Ŀ�꣬���շ�ֵ����
	std::sort(this->AllCandidateTargets, AllCandidateTargets + candidateTargetCount, CompareCandidates);

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
inline void Searcher::SearchOneRound(unsigned short* frameData)
{
	// ���֡���Ƿ����������Ϣ�������ã�
	if (CheckIfHaveGroundObject(frameIndex) == true)
	{
		frameIndex++;
		return;
	}

	// ����
	memcpy(FramesInOneRound[frameIndex], frameData, Width * Height * PixelSize);

	detector->DetectTargets(FramesInOneRound[frameIndex], &resultOfSingleFrame, nullptr, nullptr);

	CalculateScoreForDetectedTargetsAndPushToCandidateQueue(FramesInOneRound[frameIndex], resultOfSingleFrame, frameIndex);

	frameIndex++;

	if (frameIndex == FRAME_COUNT_ONE_ROUND)
	{
		GetLastResultAfterOneRound();
		candidateTargetCount = 0;
		frameIndex = 0;
	}
}

inline double Searcher::CalculateSCR(unsigned short* frame, const TargetPosition& target)
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

inline double Searcher::CalculateMaxValue(unsigned short* frame, const TargetPosition& target) const
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

inline void Searcher::CalculateScoreForDetectedTargetsAndPushToCandidateQueue(unsigned short* frame, const DetectResultSegment& result, int frameIndex)
{
	for (auto i = 0; i< result.targetCount; i++)
	{
		AllCandidateTargets[candidateTargetCount].left = result.targets[i].topLeftX;
		AllCandidateTargets[candidateTargetCount].right = result.targets[i].bottomRightX;
		AllCandidateTargets[candidateTargetCount].top = result.targets[i].topLeftY;
		AllCandidateTargets[candidateTargetCount].bottom = result.targets[i].bottomRightY;
		AllCandidateTargets[candidateTargetCount].frameIndex = frameIndex;
		AllCandidateTargets[candidateTargetCount].score = CalculateMaxValue(frame, result.targets[i]);
		candidateTargetCount++;
	}
}

#endif
