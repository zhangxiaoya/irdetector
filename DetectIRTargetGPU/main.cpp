#include "Validation/Validation.hpp"
#include "Init/Init.hpp"
#include "Validation/DetectorValidation.hpp"
#include "Network/DataReceiver.h"

#include <windows.h>
#include <iostream>
#include <mutex>
#include <thread>
#include "Models/RingBufferStruct.hpp"
#include "Models/ResultBufferStruct.hpp"

// Definition of all const varibales
extern const unsigned int WIDTH = 320;
extern const unsigned int HEIGHT = 256;
extern const unsigned int BYTESIZE = 1;

static const int BufferSize = 2;
static const int FrameSize = WIDTH * HEIGHT * BYTESIZE;

unsigned char FrameData[FrameSize];
unsigned char FrameDataInprocessing[FrameSize] = {0};
ResultSegment ResultItem;
static const int ResultItemSize = sizeof(ResultSegment);

// Init one detector
Detector* detector = new Detector();

// Definition of a ring buffer
RingBufferStruct Buffer(FrameSize, BufferSize);
ResultBufferStruct ResultBuffer(BufferSize);

/****************************************************************************************/
/*                                Input Data Operation                                  */
/****************************************************************************************/
bool InputDataToBuffer(RingBufferStruct* buffer)
{
	// Check buffer is full or not, if full automatic unlock mutex
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while ((buffer->write_position + 1) % BufferSize == buffer->read_position)
	{
		std::cout << "Producer is waiting for an empty slot...\n";
		buffer->buffer_not_full.wait(lock);
	}

	// Check finish stream end flag
	if (GetOneFrameFromNetwork(FrameData) == false)
	{
		buffer->finish_flag = true;
		return false;
	}

	// Copy data received from network to ring buffer and update ring buffer header pointer
	memcpy(buffer->item_buffer + buffer->write_position * FrameSize, FrameData, FrameSize * sizeof(unsigned char));
	buffer->write_position++;

	// Reset data header pointer when to the end of buffer
	if (buffer->write_position == BufferSize)
		buffer->write_position = 0;

	// Notify Detect thread
	buffer->buffer_not_empty.notify_all();
	lock.unlock();
	return true;
}

/****************************************************************************************/
/*                              Detect target Operation                                 */
/****************************************************************************************/
bool DetectTarget(RingBufferStruct* buffer, ResultBufferStruct* resultBuffer)
{
	// Check buffer is empty or not, if empty automatic unlock mutex
	std::unique_lock<std::mutex> readLock(buffer->bufferMutex);
	while (buffer->write_position == buffer->read_position)
	{
		if (buffer->finish_flag == true)
		{
			resultBuffer->finish_flag = true;
			return false;
		}

		std::cout << "Consumer is waiting for items...\n";
		buffer->buffer_not_empty.wait(readLock);
	}

	memcpy(FrameDataInprocessing, buffer->item_buffer + buffer->read_position * FrameSize, FrameSize);

	buffer->read_position++;

	if (buffer->read_position >= BufferSize)
		buffer->read_position = 0;

	buffer->buffer_not_full.notify_all();
	readLock.unlock();

	CheckPerf(detector->DetectTargets(FrameDataInprocessing, &ResultItem), "whole");

	// Check result buffer is full or not, if full automatic unlock mutex
//	std::unique_lock<std::mutex> writerLock(resultBuffer->bufferMutex);
//	while ((resultBuffer->write_position + 1) % BufferSize == resultBuffer->read_position)
//	{
//		std::cout << "Producer is waiting for an empty slot...\n";
//		resultBuffer->buffer_not_full.wait(writerLock);
//	}
//
//	// Copy data received from network to ring buffer and update ring buffer header pointer
//	memcpy(resultBuffer->item_buffer + resultBuffer->write_position * ResultItemSize, &ResultItem, ResultItemSize);
//	resultBuffer->write_position++;
//
//	// Reset data header pointer when to the end of buffer
//	if (resultBuffer->write_position == BufferSize)
//		resultBuffer->write_position = 0;
//
//	// Notify Detect thread
//	resultBuffer->buffer_not_empty.notify_all();
//	writerLock.unlock();
	return true;
}

/****************************************************************************************/
/*                               Send Result Operation                                  */
/****************************************************************************************/
bool OutputData(ResultBufferStruct* buffer)
{
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while (buffer->write_position == buffer->read_position)
	{
		if (buffer->finish_flag == true)
			return false;

		std::cout << "Send result thread is waiting for result items...\n";
		buffer->buffer_not_empty.wait(lock);
	}

	SendResultToRemoteServer(buffer->item_buffer[buffer->read_position]);

	buffer->read_position++;

	if (buffer->read_position >= BufferSize)
		buffer->read_position = 0;

	buffer->buffer_not_full.notify_all();
	lock.unlock();

	return true;
}

void InputDataTask()
{
	// Receiving data from network
	while (true)
	{
		if(InputDataToBuffer(&Buffer) == false)
			break;
		Sleep(1);
	}
}

void DetectTask()
{
	// Detecting target
	while (true)
	{
		if (DetectTarget(&Buffer, &ResultBuffer) == false)
			break;
	}
}

void OutputDataTask()
{
	// Send data to remote server
	while (true)
	{
		if (OutputData(&ResultBuffer) == false)
			break;
	}
}

void InitBuffer(RingBufferStruct* buffer)
{
	buffer->write_position = 0;
	buffer->read_position = 0;
}

void InitResultBuffer(ResultBufferStruct* buffer)
{
	buffer->read_position = 0;
	buffer->write_position = 0;
}


int main(int argc, char* argv[])
{
	// Init CUDA device
	auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
		// Init Network socket
//		InitNetworks();

		// Init Detector Space
//		detector->InitSpace();

		// Init ring buffer
//		InitBuffer(&Buffer);
//		InitResultBuffer(&ResultBuffer);

//		std::thread InputDataThread(InputDataTask);
//		std::thread DetectorThread(DetectTask);

//		InputDataThread.join();
//		DetectorThread.join();

		// Destroy Network
//		DestroyNetWork();

//				Validation validation;
//				validation.InitValidationData("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
//				validation.VailidationAll();

				DetectorValidation visualEffectValidator;
				visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
//				visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_2.bin");
				visualEffectValidator.VailidationAll();
	}

	// Destroy detector
	delete detector;

	// Release Cuda device
	CUDAInit::cudaDeviceRelease();

	system("Pause");
	return 0;
}
