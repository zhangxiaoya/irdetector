#ifndef __BINARY_FILE_READER__
#define __BINARY_FILE_READER__

#include "../Headers/GlobalMainHeaders.h"
#include <fstream>
#include <cuda_runtime_api.h>

class BinaryFileReader
{
public:
	BinaryFileReader(const int width, const int height, const int pixelSize, const std::string& binary_file_name)
		: frameCount(0),
		  dataMatrix(nullptr),
		  binaryFileName(binary_file_name),
		  Width(width),
		  Height(height),
		  PixelSize(pixelSize)
	{
	}

	~BinaryFileReader();

	void ReleaseSpace();

	bool ReadBinaryFileToHostMemory();

	void SetFileName(const std::string fileName);

	unsigned int GetFrameCount() const;

	unsigned short** GetDataPoint() const;

protected:
	void OpenBinaryFile(std::ifstream& fin) const;

	void CalculateFrameCount(std::ifstream& fin);

	bool InitSpaceOnHost();

	static void ConstitudePixel(unsigned char highPart, unsigned char lowPart, unsigned short& perPixel);

	void ChangeRows(unsigned& row, unsigned& col) const;

private:
	unsigned int frameCount;
	unsigned short** dataMatrix;

	std::string binaryFileName;

	int Width;
	int Height;
	int PixelSize;
};

inline BinaryFileReader::~BinaryFileReader()
{
	ReleaseSpace();
}

inline void BinaryFileReader::ReleaseSpace()
{
	if (dataMatrix != nullptr)
	{
		for (unsigned i = 0; i < frameCount; ++i)
		{
			cudaFreeHost(dataMatrix[i]);
		}
		delete[] dataMatrix;
		dataMatrix = nullptr;
	}
}

inline bool BinaryFileReader::ReadBinaryFileToHostMemory()
{
	// create one binary file reader
	std::ifstream fin;
	OpenBinaryFile(fin);

	if (fin.is_open())
	{
		// counting frame and init space on host and device respectly
		CalculateFrameCount(fin);

		// init space on host
		auto init_space_on_host = InitSpaceOnHost();
		if (init_space_on_host)
		{
			auto originalPerFramePixelArray = new unsigned short[Width * Height];
			auto iterationText = new char[200];

			// init some variables
			unsigned row = 0;          // current row index
			unsigned col = 0;          // current col index
			auto byteIndex = 2;        // current byte index
			auto frameIndex = 0;       // current frame index
			auto pixelIndex = 0;       // current pixel index

			unsigned char highPart = fin.get();
			unsigned char lowPart = fin.get();

			// main loop to read and load binary file per frame
			while (true)
			{
				// check if is the end of binary file
				if (!fin)
					break;

				// per frame
				while (byteIndex - 2 < Width * Height * PixelSize)
				{
					// take 16-bit space per pixel
					unsigned short perPixel;
					ConstitudePixel(highPart, lowPart, perPixel);

					dataMatrix[frameIndex][pixelIndex++] = perPixel;

					// update these variables
					ChangeRows(row, col);
					highPart = fin.get();
					lowPart = fin.get();
					byteIndex += 2;
				}

				sprintf_s(iterationText, 200, "Current frame index is %04d", frameIndex);
				logPrinter.PrintLogs(iterationText, LogLevel::Info);

				// prepare for next frame
				frameIndex++;
				row = 0;
				col = 0;
				byteIndex = 2;
				pixelIndex = 0;
			}
			// clean up temparory array
			if (originalPerFramePixelArray != nullptr)
			{
				delete[] originalPerFramePixelArray;
				originalPerFramePixelArray = nullptr;
			}
			if (iterationText != nullptr)
			{
				delete[] iterationText;
				iterationText = nullptr;
			}
			fin.close();
			return true;
		}
		fin.close();
		return false;
	}
	// if open binary file failed!
	logPrinter.PrintLogs("Open binary file failed, please check file path!", LogLevel::Error);
	return false;
}

inline void BinaryFileReader::SetFileName(const std::string fileName)
{
	binaryFileName = fileName;
}

inline unsigned BinaryFileReader::GetFrameCount() const
{
	return frameCount;
}

inline unsigned short** BinaryFileReader::GetDataPoint() const
{
	return dataMatrix;
}

inline void BinaryFileReader::OpenBinaryFile(std::ifstream& fin) const
{
	fin = std::ifstream(binaryFileName, std::fstream::binary | std::fstream::in);
}

inline void BinaryFileReader::CalculateFrameCount(std::ifstream& fin)
{
	logPrinter.PrintLogs("Start binary file reading ...", LogLevel::Info);
	fin.seekg(0, std::ios::end);
	const auto len = fin.tellg();

	frameCount = static_cast<unsigned>(len * 1.0 / (Width * Height * PixelSize));

	fin.seekg(0, std::ios::beg);
	logPrinter.PrintLogs("The image count in this binary file is ", LogLevel::Info, frameCount);
}

inline bool BinaryFileReader::InitSpaceOnHost()
{
	logPrinter.PrintLogs("Start init space on host ...", Info);
	// point array of all frames is on pageable memory
	dataMatrix = new unsigned short*[frameCount];

	// frame data of each frame is on pinned memory
	for (unsigned i = 0; i < frameCount; ++i)
	{
		const auto cuda_error = cudaMallocHost(reinterpret_cast<void**>(&dataMatrix[i]), Width * Height * sizeof(unsigned short));
		if (cuda_error != cudaSuccess)
		{
			logPrinter.PrintLogs("Init space on host failed! Starting roll back ...", Error);

			for (auto j = i - 1; j >= 0; j--)
				cudaFreeHost(dataMatrix[j]);

			delete[] dataMatrix;

			logPrinter.PrintLogs("Roll back done!", Info);
			return false;
		}
	}
	logPrinter.PrintLogs("Init space on host success!", Info);
	return true;
}

inline void BinaryFileReader::ConstitudePixel(unsigned char highPart, unsigned char lowPart, unsigned short& perPixel)
{
	perPixel = static_cast<unsigned short>(highPart);
	perPixel = perPixel << 8;
	perPixel |= lowPart;
}

inline void BinaryFileReader::ChangeRows(unsigned& row, unsigned& col) const
{
	col++;
	if (col == Width)
	{
		col = 0;
		row++;
	}
}
#endif
