#pragma once
#include <condition_variable>
#include "ResultSegment.hpp"

struct DetectResultRingBufferStruct
{
	explicit DetectResultRingBufferStruct(const int width, const int height, const int buffer_size)
		: finish_flag(false),
		  item_buffer(nullptr),
		  frame_buffer(nullptr),
		  read_position(0),
		  write_position(0),
		  Width(width),
		  Height(height),
		  bufferSize(buffer_size),
		  resultItemSize(0)
	{
		resultItemSize = sizeof(ResultSegment);
		item_buffer = new ResultSegment[bufferSize * resultItemSize];
		frame_buffer = new unsigned short[buffer_size * sizeof(unsigned short) * Width * Height];
	}

	~DetectResultRingBufferStruct()
	{
		delete[] item_buffer;
		delete[] frame_buffer;
	}

	bool finish_flag;
	ResultSegment* item_buffer;                      // 环形缓冲
	unsigned short* frame_buffer;                    // 帧数据缓冲区
	size_t read_position;
	size_t write_position;
	std::mutex bufferMutex;
	std::condition_variable buffer_not_full;        // 条件变量, 指示产品缓冲区不为满.
	std::condition_variable buffer_not_empty;       // 条件变量, 指示产品缓冲区不为空.

	unsigned Width;
	unsigned Height;

private:
	int bufferSize;
	int resultItemSize;
};
