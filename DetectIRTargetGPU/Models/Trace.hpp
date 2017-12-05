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

	// �����
	void EnterQueue(TraceItem _item)
	{
		TraceQueue[queueEnd] = _item;
		queueEnd++;
		// ������,ɾ������ͷԪ��
		if (queueEnd == queueFront)
		{
			queueFront++;
		}
	}

	// ������
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