#pragma once
#include <condition_variable>
#include "ResultSegment.hpp"

struct ShowResultRingBufferStruct
{
	explicit ShowResultRingBufferStruct(int _width = 320, int _height = 256, int buffer_size = 10)
		: finish_flag(false),
		  item_buffer(nullptr),
		  frame_buffer(nullptr),
		  read_position(0),
		  write_position(0),
		  width(_width),
		  height(_height),
		  bufferSize(buffer_size),
		  resultItemSize(0)
	{
		resultItemSize = sizeof(ResultSegment);
		item_buffer = new ResultSegment[bufferSize * resultItemSize];
		frame_buffer = new unsigned short[bufferSize * width * height];
	}

	~ShowResultRingBufferStruct()
	{
		delete[] item_buffer;
		delete[] frame_buffer;
	}

	bool finish_flag;
	ResultSegment* item_buffer;                      // ���λ���
	unsigned short* frame_buffer;                    // ֡���ݻ���
	size_t read_position;
	size_t write_position;
	std::mutex bufferMutex;
	std::condition_variable buffer_not_full;        // ��������, ָʾ��Ʒ��������Ϊ��.
	std::condition_variable buffer_not_empty;       // ��������, ָʾ��Ʒ��������Ϊ��.
	int width;
	int height;

private:
	int bufferSize;
	int resultItemSize;
};
