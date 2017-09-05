
#ifndef __SPLIT_BINARY_FILE__
#define __SPLIT_BINARY_FILE__

#include <fstream>
#include <iostream>

class SplitBinaryFileOperator
{
public:
	explicit SplitBinaryFileOperator(int width, int height, int count = 10);

	~SplitBinaryFileOperator();

	void Split(unsigned char highpart, unsigned char lowPart);

	bool IsReady() const;

	bool IsFinished() const;

private:
	std::ofstream fout;
	int pixelCount;
	int currentIndex;
	bool readyStatus;
	bool finshStatus;
};

inline SplitBinaryFileOperator::SplitBinaryFileOperator(int width, int height, int count)
	: pixelCount(count * width * height),
	  currentIndex(0),
	  readyStatus(false),
	  finshStatus(false)
{
	fout.open("C:\\D\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin", std::ofstream::out | std::ofstream::binary);
	if (!fout.is_open())
	{
		std::cout << "Open Splited binary file failed" << std::endl;
	}
	else
	{
		readyStatus = true;
	}
}

inline SplitBinaryFileOperator::~SplitBinaryFileOperator()
{
	if(fout.is_open())
		fout.close();
}

inline void SplitBinaryFileOperator::Split(unsigned char highpart, unsigned char lowPart)
{
	if(readyStatus && currentIndex <= pixelCount)
	{
		fout << highpart << lowPart;
		currentIndex++;
	}
	else
	{
		finshStatus = true;
	}
}

inline bool SplitBinaryFileOperator::IsReady() const
{
	return readyStatus;
}

inline bool SplitBinaryFileOperator::IsFinished() const
{
	return finshStatus;
}

#endif
