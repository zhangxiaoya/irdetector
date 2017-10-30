#ifndef __MONITOR_H__
#define __MONITOR_H__
#include "../Models/Confidences.hpp"

class Monitor
{
public:
	Monitor(const int width, const int height): currentFrame(nullptr),
	                                            Width(width),
	                                            Height(height),
	                                            confidences(nullptr)
	{
		confidences = new Confidences(Width, Height);
	}

	~Monitor()
	{
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
};

inline bool Monitor::Process(unsigned short* frame)
{

	return true;
}

inline void Monitor::InitMonitor()
{
	confidences->InitConfidenceMap();
}

#endif
