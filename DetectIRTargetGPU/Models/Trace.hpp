#pragma
#include "../Headers/SearcherParameters.h"
#include "TraceItem.hpp"

#ifndef MAX_TRACE_LIFE_TIME
#define MAX_TRACE_LIFE_TIME (10)
#endif // !MAX_TRACE_LIFE_TIME

struct Trace
{
public:
	Trace() :
		queueFront(0),
		queueEnd(0),
		LifeTime(0),
		FrameIndex(-1),
		InterIndex(-1)
	{
	}

	// 入队列
	void EnterQueue(TraceItem _item)
	{
		TraceQueue[queueEnd] = _item;
		queueEnd++;

		// 队列满,删除队列头元素
		if (TraceQueueIsFull())
		{
			queueFront++;
		}
	}

	// 出队列
	void OutQueue()
	{
		if (TraceQueueIsEmpty())
		{
			ShrinkLifeTime();
			return;
		}
		else
		{
			queueFront++;
		}
	}

	// 延长Trace生命时间
	void ExtendLifeTime()
	{
		LifeTime++;
		if (LifeTime > MAX_TRACE_LIFE_TIME)
		{
			LifeTime = MAX_TRACE_LIFE_TIME;
		}
	}

	// 缩减Trace生命时间
	void ShrinkLifeTime()
	{
		LifeTime--;
		if (LifeTime < 0)
		{
			LifeTime = 0;
		}
	}

	// 获取轨迹的长度
	int TraceLength()
	{
		return (queueEnd - queueFront + MAX_TRACE_QUEUE_LENGTH + 1) % (MAX_TRACE_QUEUE_LENGTH + 1);
	}

	// 判断Trace队列是否为空
	bool TraceQueueIsEmpty()
	{
		return queueEnd == queueFront;
	}

	// 判断Trace队列是否满
	bool TraceQueueIsFull()
	{
		return (queueFront + 1) % (MAX_TRACE_QUEUE_LENGTH + 1) == queueEnd;
	}

private:
	int queueFront;
	int queueEnd;
	int LifeTime;

	int FrameIndex;
	int InterIndex;

	TraceItem TraceQueue[MAX_TRACE_QUEUE_LENGTH + 1];
};