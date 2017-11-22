#include "Validation/CorrectnessValidation.hpp"
#include "Init/Init.hpp"
#include "Validation/PerformanceValidation.hpp"
#include "Network/NetworkTransfer.h"

#include <windows.h>
#include <iostream>
#include <mutex>
#include <thread>
#include "Models/FrameDataRingBufferStruct.hpp"
#include "Models/DetectResultRingBufferStruct.hpp"
#include "Validation/Validation.h"
#include "Monitor/Monitor.hpp"

const bool IsSendResultToServer = true; // �Ƿ��ͽ���������(������)

/****************************************************************************************/
/* �������壺 ͼ����Ϣȫ�ֱ��������붨��                                                   */
/****************************************************************************************/
extern const unsigned int WIDTH = 320;   // ͼ����
extern const unsigned int HEIGHT = 256;  // ͼ��߶�
extern const unsigned int BYTESIZE = 2;  // ÿ�������ֽ���

/****************************************************************************************/
/* �������壺Ԥ����׶β���                                                               */
/****************************************************************************************/
const int DilationRadius = 1;            // �˲����뾶
const int DiscretizationScale = 15;      // ��ɢ���߶�

/****************************************************************************************/
/* ������������                                                                          */
/****************************************************************************************/
static const int FrameDataSize = WIDTH * HEIGHT * BYTESIZE;        // ÿ��ͼ��֡���ݴ�С
static const int ImageSize = WIDTH * HEIGHT;                       // ÿһ֡ͼ�����ش�С
unsigned char FrameData[FrameDataSize];                            // ÿһ֡ͼ����ʱ����
unsigned short FrameDataInprocessing[ImageSize] = {0};             // ÿһ֡ͼ����ʱ����
unsigned short FrameDataToShow[ImageSize] = {0};                   // ÿһ֡��ʾ���ͼ����ʱ����
DetectResultSegment ResultItemSendToServer;                              // ÿһ֡ͼ������
DetectResultSegment ResultItemToShow;                                    // ÿһ֡ͼ����ʾ���
static const int ResultItemSize = sizeof(DetectResultSegment);           // ÿһ֡ͼ��������С
Detector* detector = new Detector(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);  // ��ʼ�������
Monitor* monitor = new Monitor(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);     // init monitor
cv::Mat CVFrame(HEIGHT, WIDTH, CV_8UC1);

/****************************************************************************************/
/* �������壺������ȫ�ֱ��������붨��                                                      */
/****************************************************************************************/
static const int BufferSize = 10;                                              // �߳�ͬ����������С
FrameDataRingBufferStruct Buffer(FrameDataSize, BufferSize);                   // ���ݽ����̻߳��λ�������ʼ��
DetectResultRingBufferStruct ResultBuffer(WIDTH, HEIGHT, BufferSize);          // ��������̻߳��λ�������ʼ��

/****************************************************************************************/
/* �������壺�������ȡ���ݲ�������ȡһ֡��                                                 */
/****************************************************************************************/
bool InputDataToBuffer(FrameDataRingBufferStruct* buffer)
{
	// Check buffer is full or not, if full automatic unlock mutex
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while ((buffer->write_position + 1) % BufferSize == buffer->read_position)
	{
		std::cout << "Image Data Producer is waiting for an empty slot...\n";
		buffer->buffer_not_full.wait(lock);
	}

	// Check finish stream end flag
	if (GetOneFrameFromNetwork(FrameData) == false)
	{
		buffer->finish_flag = true;
		return false;
	}

	// Copy data received from network to ring buffer and update ring buffer header pointer
	memcpy(buffer->item_buffer + buffer->write_position * FrameDataSize, FrameData, FrameDataSize);
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
/* �������壺���һ֡���ݣ����Ұѽ�����ڻ�����                                             */
/****************************************************************************************/
bool DetectTarget(FrameDataRingBufferStruct* buffer, DetectResultRingBufferStruct* resultBuffer)
{
	// ������ȡͼ������
	std::unique_lock<std::mutex> readLock(buffer->bufferMutex);
	while (buffer->write_position == buffer->read_position)
	{
		if (buffer->finish_flag == true)
		{
			resultBuffer->finish_flag = true;
			return false;
		}

		std::cout << "Detector Consumer is waiting for items...\n";
		buffer->buffer_not_empty.wait(readLock);
	}

	memcpy(FrameDataInprocessing, buffer->item_buffer + buffer->read_position * FrameDataSize, FrameDataSize);

	buffer->read_position++;

	if (buffer->read_position >= BufferSize)
		buffer->read_position = 0;

	buffer->buffer_not_full.notify_all();
	readLock.unlock();

	// ���Ŀ�꣬���������
	CheckPerf(detector->DetectTargets(FrameDataInprocessing, &ResultItemSendToServer), "Total process");
//	CheckPerf(monitor->Process(FrameDataInprocessing, &ResultItemSendToServer), "Total Tracking Process");

//	LARGE_INTEGER t1, t2, tc;
//	QueryPerformanceFrequency(&tc);
//	QueryPerformanceCounter(&t1);
//	detector->DetectTargets(FrameDataInprocessing, &ResultItemSendToServer);
//	QueryPerformanceCounter(&t2);
//	const auto timeC = (t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart;
//	printf("Operation of %20s Use Time:%f\n", "Total process", timeC);
//	if (timeC >= 0.006)
//		printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

	if(IsSendResultToServer)
	{
		// �����洢�������������
		std::unique_lock<std::mutex> writerLock(resultBuffer->bufferMutex);
		while ((resultBuffer->write_position + 1) % BufferSize == resultBuffer->read_position)
		{
			std::cout << "Result Send Producer is waiting for an empty slot...\n";
			resultBuffer->buffer_not_full.wait(writerLock);
		}

		// Copy data received from network to ring buffer and update ring buffer header pointer
		memcpy(resultBuffer->item_buffer + resultBuffer->write_position * ResultItemSize, &ResultItemSendToServer, ResultItemSize);
		resultBuffer->write_position++;

		// Reset data header pointer when to the end of buffer
		if (resultBuffer->write_position == BufferSize)
			resultBuffer->write_position = 0;

		// Notify Detect thread
		resultBuffer->buffer_not_empty.notify_all();
		writerLock.unlock();
	}

	// ��ʱ��ʾ���
	ShowFrame::ToMat(FrameDataInprocessing, WIDTH, HEIGHT, CVFrame);
	ShowFrame::DrawRectangles(CVFrame, &ResultItemSendToServer);
	cv::imshow("Result", CVFrame);
	cv::waitKey(1);

	// ����һ���߳�ִ��״̬
	return true;
}

/****************************************************************************************/
/* �������壺����һ֡�����                                                             */
/****************************************************************************************/
bool OutputData(DetectResultRingBufferStruct* buffer)
{
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while (buffer->write_position == buffer->read_position)
	{
		if (buffer->finish_flag == true)
			return false;

		std::cout << "Result send thread is waiting for result items...\n";
		buffer->buffer_not_empty.wait(lock);
	}

	memcpy(&ResultItemSendToServer, buffer->item_buffer + buffer->read_position * ResultItemSize, ResultItemSize);
	SendResultToRemoteServer(ResultItemSendToServer);

	buffer->read_position++;

	if (buffer->read_position >= BufferSize)
		buffer->read_position = 0;

	buffer->buffer_not_full.notify_all();
	lock.unlock();

	return true;
}

/****************************************************************************************/
/* �������壺��ȡͼ��֡�߳�����                                                           */
/****************************************************************************************/
void InputDataTask()
{
	// ѭ����������
	while (true)
	{
		if(InputDataToBuffer(&Buffer) == false) break;
	}
}

/****************************************************************************************/
/* �������壺���Ŀ���߳�����                                                             */
/****************************************************************************************/
void DetectTask()
{
	// ѭ����ȡͼ�񣬼��Ŀ��
	while (true)
	{
		if (DetectTarget(&Buffer, &ResultBuffer) == false) break;
	}
}

/****************************************************************************************/
/* �������壺���ͽ���߳�����                                                             */
/****************************************************************************************/
void OutputDataTask()
{
	// ѭ�����ͼ����
	while (true)
	{
		if (OutputData(&ResultBuffer) == false) break;
	}
}

/****************************************************************************************/
/* �������壺��ʼ���������ݻ���                                                            */
/****************************************************************************************/
void InitDataSourceBuffer(FrameDataRingBufferStruct* buffer)
{
	buffer->write_position = 0;
	buffer->read_position = 0;
}

/****************************************************************************************/
/* �������壺��ʼ�����������                                                           */
/****************************************************************************************/
void InitResultBuffer(DetectResultRingBufferStruct* buffer)
{
	buffer->read_position = 0;
	buffer->write_position = 0;
}

/****************************************************************************************/
/* �������壺�����绷������                                                               */
/****************************************************************************************/
void RunOnNetwork()
{
	// ��ʼ��Socket���绷��
	InitNetworks();

	// ��ʼ������Ӿֲ��洢�ͼ�����
	detector->InitSpace();
	detector->SetRemoveFalseAlarmParameters(true, false, false, false, true, true);

	// ��ʼ�����ݻ���ͽ������
	InitDataSourceBuffer(&Buffer);
	InitResultBuffer(&ResultBuffer);

	// ���������̣߳���ȡ�����̡߳������������ؽ��
	std::thread InputDataThread(InputDataTask);
	std::thread DetectorThread(DetectTask);
	std::thread OutputDataThread(OutputDataTask);

	// �����߳̿�ʼ����
	InputDataThread.join();
	DetectorThread.join();
	OutputDataThread.join();

	// ��������
	DestroyNetWork();
}

/****************************************************************************************/
/*                                     Main Function                                    */
/****************************************************************************************/
int main(int argc, char* argv[])
{
	// ��ʼ��CUDA�豸
	const auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
//		RunOnNetwork();

//		CheckConrrectness(WIDTH, HEIGHT);

//		CheckPerformance(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);

//		CheckTracking(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);

		CheckSearching(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);
	}

	// ���ټ����
	delete detector;
	delete monitor;

	// �ͷ�CUDA�豸
	CUDAInit::cudaDeviceRelease();

	// ϵͳ��ͣ
	system("Pause");
	return 0;
}
