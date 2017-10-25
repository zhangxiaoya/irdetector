#ifndef __MONITOR_H__
#define __MONITOR_H__
#include <cstring>
#include <cuda_runtime_api.h>

// To-Do
class Monitor
{
public:
	Monitor(const int width, const int height):
		originalFrameOnHost(nullptr),
		originalFrameOnDevice(nullptr),
		prepareResultOnHost(nullptr),
		prepareResultOnDevice(nullptr),
		Width(width),
		Height(height)
	{
	}

	void ResetData(unsigned char* dataSource) const;

	void Dialation();

private:
	unsigned char* originalFrameOnHost;
	unsigned char* originalFrameOnDevice;
	unsigned char* prepareResultOnHost;
	unsigned char* prepareResultOnDevice;

	int Width;
	int Height;
};

inline void Monitor::ResetData(unsigned char* dataSource) const
{
	if(dataSource !=nullptr)
	{
		memcpy(this->originalFrameOnHost, dataSource, sizeof(unsigned char) * Width * Height);
		cudaMemcpy(originalFrameOnDevice, originalFrameOnHost, sizeof(unsigned char) * Width * Height, cudaMemcpyHostToDevice);
	}
}

inline void Monitor::Dialation()
{
	// To-Do
}

#endif
