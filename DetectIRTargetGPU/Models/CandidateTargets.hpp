#pragma once
struct CandidateTarget
{
	int top;
	int bottom;
	int left;
	int right;
	int frameIndex;
	double score;

	CandidateTarget() : top(-1), bottom(-1), left(-1), right(-1), frameIndex(-1), score(0.0)
	{
	}
};
