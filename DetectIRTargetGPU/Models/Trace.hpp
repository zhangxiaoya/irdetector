#pragma
#include "../Headers/SearcherParameters.h"
#include "TraceItem.hpp"

struct Trace
{
public:
	Trace() :
		queueFront(0),
		queueEnd(0),
		LifeTime(0)
	{
	}

	// 入队列
	void EnterQueue(TraceItem _item)
	{
		TraceQueue[queueEnd] = _item;
		queueEnd++;
		// 队列满,删除队列头元素
		if (queueEnd == queueFront)
		{
			queueFront++;
		}
	}

	// 出队列
	void OutQueue()
	{
		if (queueFront == queueEnd)
		{
			return;
		}
		else
		{
			queueFront++;
		}
	}

private:
	int queueFront;
	int queueEnd;
	int LifeTime;
	TraceItem TraceQueue[MAX_TRACE_QUEUE_LENGTH + 1];
};