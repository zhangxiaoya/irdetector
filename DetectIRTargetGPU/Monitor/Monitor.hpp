#ifndef __MONITOR_H__
#define __MONITOR_H__
#include "../Models/Confidences.hpp"
#include "../Detector/Detector.hpp"

const int ConfValue = 12;

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
		InitDetector();
		InitMonitor();
	}

	~Monitor()
	{
		delete detector;
		delete confidences;
	}

	bool Process(unsigned short* frame);

protected:
	void InitMonitor();

	void InitDetector();

private:
	unsigned short* currentFrame;
	int Width;
	int Height;
	int DilationRadius;
	int DiscretizationScale;

	Confidences* confidences;
	Detector* detector;
	ResultSegment result;
};

inline bool Monitor::Process(unsigned short* frame)
{
	detector->DetectTargets(frame, &result);

	for (auto i = 0; i < result.targetCount; ++i)
	{
		const auto centerX = result.targets[i].bottomRightX + result.targets[i].topLeftX;
		const auto centerY = result.targets[i].bottomRightY + result.targets[i].topLeftY;
		const auto BlockX = centerX / BlockSize;
		const auto BlockY = centerY / BlockSize;

		confidences->ConfidenceMap[BlockY][BlockX][confidences->QueueEnd] = ConfValue;
	}
	confidences->QueueEnd++;
	if (confidences->QueueBeg == confidences->QueueEnd)
		confidences->QueueBeg++;

	return true;
}

inline void Monitor::InitMonitor()
{
	this->confidences = new Confidences(Width, Height);
	this->confidences->InitConfidenceMap();
}

inline void Monitor::InitDetector()
{
	this->detector = new Detector(Width, Height, DilationRadius, DiscretizationScale);
}

#endif
