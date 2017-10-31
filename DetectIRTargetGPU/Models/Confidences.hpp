#ifndef __CONFIDEBCES_H__
#define __CONFIDEBCES_H__

const int BlockSize = 16;

typedef int ConfElem[6];

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

	ConfElem* ConfidenceMap;
};

inline Confidences::~Confidences()
{
	delete[]  ConfidenceMap;
}

inline void Confidences::InitConfidenceMap()
{
	this->ConfidenceMap = new ConfElem[BlockRows * BlockCols];
	QueueBeg = QueueEnd = 0;
}

#endif
