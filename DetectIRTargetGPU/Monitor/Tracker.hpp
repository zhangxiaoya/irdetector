#ifndef __TRACKER_H__
#define __TRACKER_H__
#include "../Models/DetectResultSegment.hpp"

const int MaxLifeTime = 5;

class Tracker
{
public:
	Tracker();

	void InitLifeTime();

	void ExtendLifeTime();

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

inline void Tracker::InitLifeTime()
{
	LifeTime = 1;
}

inline void Tracker::ExtendLifeTime()
{
	++LifeTime;
	if (LifeTime > MaxLifeTime)
		LifeTime = MaxLifeTime;
}

inline void Tracker::ShrinkLifeTime()
{
	--LifeTime;
	if (LifeTime <= 0)
		this->ValidFlag = false;
}

#endif
