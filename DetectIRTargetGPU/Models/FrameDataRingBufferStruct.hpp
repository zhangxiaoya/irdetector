#pragma once
#include <condition_variable>

struct FrameDataRingBufferStruct
{
	explicit FrameDataRingBufferStruct(int frame_size, int buffer_size = 10)
		: finish_flag(false),
		  item_buffer(nullptr),
		  read_position(0),
		  write_position(0),
		  bufferSize(buffer_size),
		  frameSize(frame_size)
	{
		item_buffer = new unsigned char[bufferSize * frameSize];
	}

	~FrameDataRingBufferStruct()
	{
		delete[] item_buffer;
	}

	bool finish_flag;
	unsigned char* item_buffer;                     // 环形缓冲
	size_t read_position;
	size_t write_position;
	std::mutex bufferMutex;
	std::condition_variable buffer_not_full;        // 条件变量, 指示产品缓冲区不为满.
	std::condition_variable buffer_not_empty;       // 条件变量, 指示产品缓冲区不为空.

private:
	int bufferSize;
	int frameSize;
};
