#ifndef __LAZY_DETECTOR_H__

#include "Detector.hpp"
/*****************************************************************************************************************************/
/* ÑÓ³ÙËÑË÷Àà¶¨Òå                                                                                                            */
/*****************************************************************************************************************************/
class LazyDetector
{
public:
	LazyDetector(const int width, const int height, const int dilationRadius, const int discretizationScale);

	~LazyDetector();

	void InitDetector();

	void DetectTargets(unsigned short* frame, DetectResultSegment* result);

private:
	Detector* detector;

	int Width;
	int Height;
	int DialationRadius;
	int DiscretizationScale;
};

LazyDetector::LazyDetector(const int width, const int height, const int dilationRadius, const int discretizationScale)
	:detector(nullptr),
	 Width(width),
	 Height(height),
	 DialationRadius(dilationRadius),
	 DiscretizationScale(discretizationScale)
{
}

LazyDetector::~LazyDetector()
{
	delete detector;
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
}
#endif // !__LAZY_DETECTOR_H__
