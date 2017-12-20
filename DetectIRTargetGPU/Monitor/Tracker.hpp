#ifndef __TRACKER_H__
#define __TRACKER_H__
#include "../Models/DetectResultSegment.hpp"

const int MaxLifeTime = 5;

class Tracker
{
public:
	Tracker();

	void InitLifeTime(Point& centerPos);

	void ExtendLifeTime(Point& centerPos);

	void ShrinkLifeTime();

	bool ValidFlag;
	int LifeTime;
	TargetPosition Postion;
	unsigned short Area;
	int BlockX;
	int BlockY;

	Point CenterList[MaxLifeTime + 1];
	int posBeg;
	int posEnd;
};

inline Tracker::Tracker(): ValidFlag(false), LifeTime(0), BlockX(0), BlockY(0), Area(0), posBeg(0), posEnd(0)
{
}

inline void Tracker::InitLifeTime(Point& centerPos)
{
	LifeTime = 1;
	CenterList[posEnd] = centerPos;
	posEnd++;
}

inline void Tracker::ExtendLifeTime(Point& centerPos)
{
	++LifeTime;
	if (LifeTime > MaxLifeTime)
		LifeTime = MaxLifeTime;

	CenterList[posEnd] = centerPos;
	posEnd++;
	if (posEnd == (MaxLifeTime + 1))
		posEnd = 0;
	if ((posEnd + 1) % (MaxLifeTime + 1) == posBeg)
	{
		posBeg++;
		if (posBeg == (MaxLifeTime + 1))
			posBeg = 0;
	}
}

inline void Tracker::ShrinkLifeTime()
{
	--LifeTime;
	if (LifeTime <= 0)
	{
		this->ValidFlag = false;
		posBeg = 0;
		posEnd = 0;
	}
}

#endif
