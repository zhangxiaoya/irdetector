#ifndef __MONITOR_H__
#define __MONITOR_H__
#include <cstring>
#include <cuda_runtime_api.h>

// To-Do
class Monitor
{
public:
	Monitor(): originalFrameOnHost(nullptr), originalFrameOnDevice(nullptr), prepareResultOnHost(nullptr), prepareResultOnDevice(nullptr), width(320), height(256)
	{
	}

	void ResetData(unsigned char* dataSource) const;

	void Dialation();

private:
	unsigned char* originalFrameOnHost;
	unsigned char* originalFrameOnDevice;
	unsigned char* prepareResultOnHost;
	unsigned char* prepareResultOnDevice;

	int width;
	int height;
};

inline void Monitor::ResetData(unsigned char* dataSource) const
{
	if(dataSource !=nullptr)
	{
		memcpy(this->originalFrameOnHost, dataSource, sizeof(unsigned char) * width * height);
		cudaMemcpy(originalFrameOnDevice, originalFrameOnHost, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);
	}
}

inline void Monitor::Dialation()
{
	// To-Do
}

#endif
