#pragma once
#include "../DataReaderFromeFiles/BinaryFileReader.hpp"
#include "../Models/LogLevel.hpp"
#include "../LogPrinter/LogPrinter.hpp"
#include "../Dilations/DilatetionOnCPU.hpp"
#include "../Dilations/DilatetionKernel.h"
#include "../Checkers/CheckDiff.hpp"
#include "../LevelDiscretization/LevelDiscretizationOnCPU.hpp"
#include "../LevelDiscretization/LevelDiscretizationKernel.cuh"
#include "../Checkers/CheckCUDAReturnStatus.h"

class Validation
{
public:
	explicit Validation(BinaryFileReader* file_reader = nullptr)
		: fileReader(file_reader),
		  originalFrameOnHost(nullptr),
		  originalFrameOnDeivce(nullptr),
		  resultOfDilationOnHostUseCPU(nullptr),
		  resultOfDilationOnHostUseGPU(nullptr),
		  resultOfDilationOnDevice(nullptr),
		  resultOfLevelDiscretizationOnHostUseCPU(nullptr),
		  resultOfLevelDiscretizationOnHostUseGPU(nullptr),
		  resultOfLevelDiscretizationOnDevice(nullptr),
		  width(320),
		  height(256)
	{
	}

	~Validation()
	{
		delete fileReader;
		DestroySpace();
	}

	void InitValidationData(std::string validationFileName);

	bool VailidationAll();

protected:
	bool DilationValidation() const;

	bool LevelDiscretizationValidation() const;

	void InitSpace() const;

	void DestroySpace() const;

private:
	BinaryFileReader* fileReader;

	unsigned char* originalFrameOnHost;
	unsigned char* originalFrameOnDeivce;
	unsigned char* resultOfDilationOnHostUseCPU;
	unsigned char* resultOfDilationOnHostUseGPU;
	unsigned char* resultOfDilationOnDevice;
	unsigned char* resultOfLevelDiscretizationOnHostUseCPU;
	unsigned char* resultOfLevelDiscretizationOnHostUseGPU;
	unsigned char* resultOfLevelDiscretizationOnDevice;

	int width;
	int height;

	LogPrinter logPrinter;
};

inline void Validation::InitValidationData(std::string validationFileName)
{
	if(fileReader != nullptr)
	{
		delete fileReader;
		fileReader = nullptr;
	}
	fileReader = new BinaryFileReader(validationFileName);
	fileReader->ReadBinaryFileToHostMemory();

	this->InitSpace();
}

inline bool Validation::VailidationAll()
{
	auto frameCount = fileReader->GetFrameCount();
	auto dataPoint = fileReader->GetDataPoint();

	auto checkResult = false;

	char iterationText[200];

	for(auto i = 0;i<frameCount;++i)
	{
		sprintf_s(iterationText, 200, "Checking for frame %04d ...", i);
		logPrinter.PrintLogs(iterationText, Info);

		memset(resultOfDilationOnHostUseCPU, 0, width * height * sizeof(unsigned char));
		memset(resultOfDilationOnHostUseGPU, 0, width * height * sizeof(unsigned char));
		memset(resultOfLevelDiscretizationOnHostUseCPU, 0, width * height * sizeof(unsigned char));
		memset(resultOfLevelDiscretizationOnHostUseGPU, 0, width * height * sizeof(unsigned char));
		cudaMemcpy(resultOfDilationOnDevice, resultOfDilationOnHostUseGPU, sizeof(unsigned char)*width * height, cudaMemcpyHostToDevice);
		cudaMemcpy(resultOfLevelDiscretizationOnDevice, resultOfLevelDiscretizationOnHostUseGPU, sizeof(unsigned char)*width * height, cudaMemcpyHostToDevice);

		originalFrameOnHost = dataPoint[i];

		checkResult = DilationValidation();
		if(checkResult == false)
			break;

		memcpy(resultOfLevelDiscretizationOnHostUseCPU, resultOfDilationOnHostUseCPU, sizeof(unsigned char) * width * height);
		cudaMemcpy(resultOfLevelDiscretizationOnDevice, resultOfDilationOnDevice, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToDevice);

		checkResult = LevelDiscretizationValidation();
		if(checkResult == false)
			break;
	}
	return checkResult;
}

inline bool Validation::DilationValidation() const
{
	cudaMemcpy(originalFrameOnDeivce, originalFrameOnHost, sizeof(unsigned char) *width * height, cudaMemcpyHostToDevice);

	logPrinter.PrintLogs("Dilation on CPU!", Info);
	DilationOnCPU::dilationCPU(originalFrameOnHost, resultOfDilationOnHostUseCPU, width, height, 1);

	logPrinter.PrintLogs("Dialtion On GPU", Info);
	FilterDilation(originalFrameOnDeivce, resultOfDilationOnDevice, width, height, 1);
	cudaMemcpy(resultOfDilationOnHostUseGPU, resultOfDilationOnDevice, width * height, cudaMemcpyDeviceToHost);

	return CheckDiff::Check(resultOfDilationOnHostUseCPU, resultOfDilationOnHostUseGPU, width, height);
}

inline bool Validation::LevelDiscretizationValidation() const
{
	logPrinter.PrintLogs("Level Discretization On CPU", Info);
	LevelDiscretizationOnCPU::LevelDiscretization(resultOfLevelDiscretizationOnHostUseCPU, width, height, 15);

	logPrinter.PrintLogs("Level Discretization On GPU", Info);
	LevelDiscretizationOnGPU(resultOfLevelDiscretizationOnDevice, width, height, 15);
	cudaMemcpy(resultOfLevelDiscretizationOnHostUseGPU, resultOfLevelDiscretizationOnDevice, width * height, cudaMemcpyDeviceToHost);

	return CheckDiff::Check(resultOfLevelDiscretizationOnHostUseCPU, resultOfLevelDiscretizationOnHostUseGPU, width, height);
}

inline void Validation::InitSpace() const
{
	CheckCUDAReturnStatus(cudaMalloc((void**)&this->originalFrameOnDeivce, sizeof(unsigned char) * width * height));
	CheckCUDAReturnStatus(cudaMalloc((void**)&this->resultOfDilationOnDevice, sizeof(unsigned char) * width * height));
	CheckCUDAReturnStatus(cudaMalloc((void**)&this->resultOfLevelDiscretizationOnDevice, sizeof(unsigned char) * height * width));

	CheckCUDAReturnStatus(cudaMallocHost((void**)&this->resultOfDilationOnHostUseCPU,sizeof(unsigned char) * width * height));
	CheckCUDAReturnStatus(cudaMallocHost((void**)&this->resultOfDilationOnHostUseGPU,sizeof(unsigned char) * width * height));
	CheckCUDAReturnStatus(cudaMallocHost((void**)&this->resultOfLevelDiscretizationOnHostUseCPU, sizeof(unsigned char) *width * height));
	CheckCUDAReturnStatus(cudaMallocHost((void**)&this->resultOfLevelDiscretizationOnHostUseGPU, sizeof(unsigned char) *width * height));
}

inline void Validation::DestroySpace() const
{
	cudaFree(this->originalFrameOnDeivce);
	cudaFree(this->resultOfDilationOnDevice);
	cudaFree(this->resultOfLevelDiscretizationOnDevice);

	cudaFreeHost(this->resultOfDilationOnHostUseCPU);
	cudaFreeHost(this->resultOfDilationOnHostUseGPU);
	cudaFreeHost(this->resultOfLevelDiscretizationOnHostUseCPU);
	cudaFreeHost(this->resultOfLevelDiscretizationOnHostUseGPU);
}
