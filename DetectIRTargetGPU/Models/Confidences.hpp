#ifndef __CONFIDEBCES_H__
#define __CONFIDEBCES_H__

#ifndef CONFIDENCE_QUEUE_ELEM_SIZE
#define CONFIDENCE_QUEUE_ELEM_SIZE 6
#endif

const int BlockSize = 10;

typedef int ConfQueueElem[CONFIDENCE_QUEUE_ELEM_SIZE];

class Confidences
{
public:
	Confidences(const int width, const int height, const int blockCols, const int blockRows):
		BlockCols(blockCols),
		BlockRows(blockRows),
		Width(width),
		Height(height),
		QueueBeg(0),
		QueueEnd(0),
		ConfidenceMap(nullptr)
	{
	}

	~Confidences();

	void InitConfidenceMap();

	int BlockCols;
	int BlockRows;
	int Width;
	int Height;

	int QueueBeg;
	int QueueEnd;

	ConfQueueElem* ConfidenceMap;
};

inline Confidences::~Confidences()
{
	delete[]  ConfidenceMap;
}

inline void Confidences::InitConfidenceMap()
{
	this->ConfidenceMap = new ConfQueueElem[BlockRows * BlockCols];
	for(auto R = 0; R < BlockRows; ++R)
	{
		for(auto C = 0; C < BlockCols; ++C)
		{
			memset(this->ConfidenceMap[R * BlockRows + C],0,sizeof(int) * CONFIDENCE_QUEUE_ELEM_SIZE);
		}
	}
	QueueBeg = QueueEnd = 0;
}

#endif
