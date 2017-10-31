#ifndef __TRACKER_H__
#define __TRACKER_H__
#include "../Models/DetectResultSegment.hpp"

class Tracker
{
public:
	Tracker();

	bool ValidFlag;
	int LifeTime;
	TargetPosition Postion;
	int BlockX;
	int BlockY;
};

inline Tracker::Tracker(): ValidFlag(false), LifeTime(0), BlockX(0), BlockY(0)
{
}

#endif
