#ifndef __TRACKER_H__
#define __TRACKER_H__
#include "../Models/DetectResultSegment.hpp"

const int MaxLifeTime = 5;

#ifndef CENTER_POSITION_DIFF
#define CENTER_POSITION_DIFF (1)
#endif // !CENTER_POSITION_DIFF


class Tracker
{
public:
	Tracker();

	void InitLifeTime(Point& centerPos);

	void ExtendLifeTime(Point& centerPos);

	void ShrinkLifeTime();

	bool IsComming();

	bool IsOutting();

	bool IsNotMove();

	bool ValidFlag;
	int LifeTime;
	TargetPosition Postion;
	TargetInfo Info;
	int BlockX;
	int BlockY;

	Point CenterList[MaxLifeTime + 1];
	int posBeg;
	int posEnd;

private:
	bool MovingCheck();

	bool HoritalMovingCheck();

	bool VartitalMovingCheck();
};

inline Tracker::Tracker(): ValidFlag(false), LifeTime(0), BlockX(0), BlockY(0), posBeg(0), posEnd(0)
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

inline bool Tracker::IsComming()
{
	return MovingCheck();
}

inline bool Tracker::IsOutting()
{
	return !MovingCheck();
}

inline bool Tracker::IsNotMove()
{
	return !VartitalMovingCheck() && !HoritalMovingCheck();
}

inline bool Tracker::MovingCheck()
{
	int positiveCount = 0;
	int i = posEnd - 1;
	if (i == -1)
		i = MaxLifeTime;
	int previousPos = 0;
	while (i != posBeg)
	{
		if (CenterList[i].x - previousPos > CENTER_POSITION_DIFF)
		{
			positiveCount++;
		}
		else
		{
			positiveCount--;
		}
		previousPos = CenterList[i].x;
		i--;
		if (i == -1)
			i = MaxLifeTime;
	}
	if (positiveCount > 1)
		return true;
	return false;
}

inline bool Tracker::HoritalMovingCheck()
{
	int positiveCount = 0;
	int i = posEnd - 1;
	if (i == -1)
		i = MaxLifeTime;
	int previousPos = 0;
	while (i != posBeg)
	{
		if (std::abs(CenterList[i].y - previousPos) > CENTER_POSITION_DIFF)
		{
			positiveCount++;
		}
		else
		{
			positiveCount--;
		}
		previousPos = CenterList[i].y;
		i--;
		if (i == -1)
			i = MaxLifeTime;
	}
	if (positiveCount > 2)
		return true;
	return false;
}

inline bool Tracker::VartitalMovingCheck()
{
	int positiveCount = 0;
	int i = posEnd - 1;
	if (i == -1)
		i = MaxLifeTime;
	int previousPos = 0;
	while (i != posBeg)
	{
		if (std::abs(CenterList[i].x - previousPos) > CENTER_POSITION_DIFF)
		{
			positiveCount++;
		}
		else
		{
			positiveCount--;
		}
		previousPos = CenterList[i].x;
		i--;
		if (i == -1)
			i = MaxLifeTime;
	}
	if (positiveCount > 2)
		return true;
	return false;
}

#endif
