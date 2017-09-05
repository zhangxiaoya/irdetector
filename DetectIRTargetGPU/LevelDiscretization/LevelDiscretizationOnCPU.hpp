#ifndef __LEVEL_DISCRETIZATION_ON_GPU__
#define __LEVEL_DISCRETIZATION_ON_GPU__

class LevelDiscretizationOnCPU
{
public:
	static void LevelDiscretization(unsigned char* frame, int width, int height, int discretizationScale);
};

inline void LevelDiscretizationOnCPU::LevelDiscretization(unsigned char* frame, int width, int height, int discretizationScale)
{
	for(auto i = 0 ;i< width * height; ++i)
	{
		frame[i] = static_cast<unsigned char>(static_cast<int>(frame[i]) / discretizationScale * discretizationScale);
	}
}

#endif
