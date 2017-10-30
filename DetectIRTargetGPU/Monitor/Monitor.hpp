#ifndef __MONITOR_H__
#define __MONITOR_H__

class Monitor
{
public:
	Monitor(const int width, const int height): currentFrame(nullptr),
	                                            Width(width),
	                                            Height(height)
	{
	}

	bool Process(unsigned short* frame);

private:

	unsigned short* currentFrame;
	int Width;
	int Height;
};

inline bool Monitor::Process(unsigned short* frame)
{

	return true;
}

#endif
