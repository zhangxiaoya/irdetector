#ifndef __LEVEL_DISCRETIZATION_ON_GPU__
#define __LEVEL_DISCRETIZATION_ON_GPU__

class LevelDiscretizationOnCPU
{
public:
	static void LevelDiscretization(unsigned short* frameOnHost, int width, int height, int discretizationScale);
};

inline void LevelDiscretizationOnCPU::LevelDiscretization(unsigned short* frameOnHost, int width, int height, int discretizationScale)
{
	for(auto i = 0 ;i< width * height; ++i)
	{
		frameOnHost[i] = static_cast<unsigned short>(static_cast<int>(frameOnHost[i]) / discretizationScale * discretizationScale);
	}
}

#endif
