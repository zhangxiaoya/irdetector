#ifndef __MONITOR_H__
#define __MONITOR_H__
#include "../Models/Confidences.hpp"
#include "../Detector/Detector.hpp"

class Monitor
{
public:
	Monitor(const int width, const int height): currentFrame(nullptr),
	                                            Width(width),
	                                            Height(height),
	                                            confidences(nullptr),
	                                            detector(nullptr)
	{
		confidences = new Confidences(Width, Height);
	}

	~Monitor()
	{
		delete detector;
		delete confidences;
	}

	bool Process(unsigned short* frame);

protected:
	void InitMonitor();

private:
	unsigned short* currentFrame;
	int Width;
	int Height;

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
		int BlockX = centerX / BlockSize;
		int BlockY = centerY / BlockSize;
	}

	return true;
}

inline void Monitor::InitMonitor()
{
	confidences->InitConfidenceMap();
}

#endif
