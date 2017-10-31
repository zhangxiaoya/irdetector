#ifndef __TRACKER_H__
#define __TRACKER_H__
#include "Models/ResultSegment.hpp"

class Tracker
{
public:
	Tracker();

	bool ValidFlag;
	int LifeTime;
	TargetPosition Postion;
};

inline Tracker::Tracker(): ValidFlag(false), LifeTime(0)
{
}

#endif
