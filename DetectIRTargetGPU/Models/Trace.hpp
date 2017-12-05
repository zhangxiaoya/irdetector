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

	// �����
	void EnterQueue(TraceItem _item)
	{
		TraceQueue[queueEnd] = _item;
		queueEnd++;

		// ������,ɾ������ͷԪ��
		if (TraceQueueIsFull())
		{
			queueFront++;
		}
	}

	// ������
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

	// �ӳ�Trace����ʱ��
	void ExtendLifeTime()
	{
		LifeTime++;
		if (LifeTime > MAX_TRACE_LIFE_TIME)
		{
			LifeTime = MAX_TRACE_LIFE_TIME;
		}
	}

	// ����Trace����ʱ��
	void ShrinkLifeTime()
	{
		LifeTime--;
		if (LifeTime < 0)
		{
			LifeTime = 0;
		}
	}

	// ��ȡ�켣�ĳ���
	int TraceLength()
	{
		return (queueEnd - queueFront + MAX_TRACE_QUEUE_LENGTH + 1) % (MAX_TRACE_QUEUE_LENGTH + 1);
	}

	// �ж�Trace�����Ƿ�Ϊ��
	bool TraceQueueIsEmpty()
	{
		return queueEnd == queueFront;
	}

	// �ж�Trace�����Ƿ���
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