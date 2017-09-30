#pragma once
#include <condition_variable>
#include "ResultSegment.hpp"

struct ShowResultRingBufferStruct
{
	explicit ShowResultRingBufferStruct(int buffer_size = 10)
		: finish_flag(false),
		item_buffer(nullptr),
		read_position(0),
		write_position(0),
		bufferSize(buffer_size),
		resultItemSize(0)
	{
		resultItemSize = sizeof(ResultSegment);
		item_buffer = new ResultSegment[bufferSize * resultItemSize];
	}

	~ShowResultRingBufferStruct()
	{
		delete[] item_buffer;
	}

	bool finish_flag;
	ResultSegment* item_buffer;                      // ���λ���
	size_t read_position;
	size_t write_position;
	std::mutex bufferMutex;
	std::condition_variable buffer_not_full;        // ��������, ָʾ��Ʒ��������Ϊ��.
	std::condition_variable buffer_not_empty;       // ��������, ָʾ��Ʒ��������Ϊ��.

private:
	int bufferSize;
	int resultItemSize;
};
