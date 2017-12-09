#include <windows.h>
#include <mutex>
#include <thread>
#include <core/core.hpp>
#include "../Headers/FrameParameters.h"
#include "../Models/DetectResultSegment.hpp"
#include "../Detector/Detector.hpp"
#include "../Headers/PreProcessParameters.h"
#include "../Monitor/Monitor.hpp"
#include "../Models/FrameDataRingBufferStruct.hpp"
#include "../Models/DetectResultRingBufferStruct.hpp"
#include "../Checkers/CheckPerf.h"
#include "../Monitor/Searcher.hpp"
#include "../Monitor/MultiSearcher.hpp"
#include "NetworkTransfer.h"
#pragma comment(lib, "winmm.lib")

/****************************************************************************************/
/* ������������                                                                          */
/****************************************************************************************/
unsigned char FrameData[FRAME_DATA_SIZE];                             // ÿһ֡ͼ����ʱ����
unsigned short FrameDataInprocessing[IMAGE_SIZE] = { 0 };             // ÿһ֡ͼ����ʱ����
unsigned short FrameDataToShow[IMAGE_SIZE] = { 0 };                   // ÿһ֡��ʾ���ͼ����ʱ����
DetectResultSegment ResultItemSendToServer;                              // ÿһ֡ͼ������
static const int ResultItemSize = sizeof(DetectResultSegment);           // ÿһ֡ͼ��������С
cv::Mat CVFrame(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);

/****************************************************************************************/
/* ���������                                                                          */
/****************************************************************************************/
Detector* detector = new Detector(IMAGE_WIDTH, IMAGE_HEIGHT, DIALATION_KERNEL_RADIUS, DISCRETIZATION_SCALE);  // ��ʼ�������
Monitor* monitor = new Monitor(IMAGE_WIDTH, IMAGE_HEIGHT, DIALATION_KERNEL_RADIUS, DISCRETIZATION_SCALE);     // ��ʼ��Monitor

Searcher* searcher = new Searcher(IMAGE_WIDTH, IMAGE_HEIGHT, PIXEL_SIZE, DIALATION_KERNEL_RADIUS, DISCRETIZATION_SCALE); // ��ʼ�������㷨
MultiSearcher* multiSearcher = new MultiSearcher(IMAGE_WIDTH, IMAGE_HEIGHT, PIXEL_SIZE, DIALATION_KERNEL_RADIUS, DISCRETIZATION_SCALE); // ��ʼ�������㷨

/****************************************************************************************/
/* �������壺������ȫ�ֱ��������붨��                                                      */
/****************************************************************************************/
FrameDataRingBufferStruct Buffer(FRAME_DATA_SIZE, RING_BUFFER_SIZE);                    // ���ݽ����̻߳��λ�������ʼ��
DetectResultRingBufferStruct ResultBuffer(IMAGE_WIDTH, IMAGE_HEIGHT, RING_BUFFER_SIZE); // ��������̻߳��λ�������ʼ��

/****************************************************************************************/
/* �������壺��ʾ����������ã�                                                           */
/****************************************************************************************/
void ShowLastResult(int shouLastResultDelay)
{
	ShowFrame::ToMat(FrameDataInprocessing, IMAGE_WIDTH, IMAGE_HEIGHT, CVFrame);
	ShowFrame::DrawRectangles(CVFrame, &ResultItemSendToServer);
	cv::imshow("Result", CVFrame);
	cv::waitKey(shouLastResultDelay);
}

/****************************************************************************************/
/* �������壺�ͷ�ָ��                                                                    */
/****************************************************************************************/
inline void RelaseSource()
{
	delete detector;
	delete monitor;
	delete searcher;
	delete multiSearcher;
}

/****************************************************************************************/
/* �������壺�������ȡ���ݲ�������ȡһ֡��                                                 */
/****************************************************************************************/
bool InputDataToBuffer(FrameDataRingBufferStruct* buffer)
{
	// Check buffer is full or not, if full automatic unlock mutex
	// ��黷�λ������Ƿ��Ѿ����ˣ�������ˣ�������������ѻ��������ݵ��̶߳�ȡ�������ڵ�����
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while ((buffer->write_position + 1) % RING_BUFFER_SIZE == buffer->read_position)
	{
		// ����ɾ����ӡ��־������û��Ҫ��ʱ������
		// printf("Image Data Producer is waiting for an empty slot...\n");
		buffer->buffer_not_full.wait(lock);
	}

	// �������ȡһ֡ͼ�����ݣ�����ȡ�Ľ���ǽ�����־��10���ֽڵ��������ݣ����޸Ļ��������־Ϊtrue����ֹ�������������߳�
	// Check finish stream end flag
	if (GetOneFrameFromNetwork(FrameData) == false)
	{
		buffer->finish_flag = true;
		return false;
	}

	// �ѽ��ܵ������ݷ��ڻ������������޸Ļ������ָ��
	// Copy data received from network to ring buffer and update ring buffer header pointer
	memcpy(buffer->item_buffer + buffer->write_position * FRAME_DATA_SIZE, FrameData, FRAME_DATA_SIZE);
	buffer->write_position++;

	// ���λ���ָ���жϣ���ָֹ�����Խ��
	// Reset data header pointer when to the end of buffer
	if (buffer->write_position == RING_BUFFER_SIZE)
		buffer->write_position = 0;

	// ֪ͨ���������̣߳��Ѿ������ݴ��ڻ�������������������
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
	// Check buffer is empty or not, if empty automatic unlock mutex
	// ��黺�����Ƿ�Ϊ�գ����Ϊ�գ������������
	std::unique_lock<std::mutex> readLock(buffer->bufferMutex);
	while (buffer->write_position == buffer->read_position)
	{
		// �ж��ǻ��������־�����ǻ�����������ʾΪtrue��������߳�
		if (buffer->finish_flag == true)
		{
			resultBuffer->finish_flag = true;
			return false;
		}
		// ע�ʹ�ӡ��־�����ⲻ��Ҫ��ʱ�俪��
		// printf("Detector Consumer is waiting for items...\n");
		buffer->buffer_not_empty.wait(readLock);
	}

	// �ӻ�������ȡ����Ҫ��������ݣ����洢����ʱͼ��洢����
	memcpy(FrameDataInprocessing, buffer->item_buffer + buffer->read_position * FRAME_DATA_SIZE, FRAME_DATA_SIZE);

	// �޸Ļ�������־
	buffer->read_position++;

	if (buffer->read_position >= RING_BUFFER_SIZE)
		buffer->read_position = 0;

	// ���λ���ָ���жϣ���ָֹ�����Խ��
	buffer->buffer_not_full.notify_all();
	readLock.unlock();

	// ���Ŀ�꣬���������
	 CheckPerf(detector->DetectTargets(FrameDataInprocessing, &ResultItemSendToServer), "Total process");
	// ��Ⲣ����Ŀ�꣬����������̵�ʱ��ϵ��
	//CheckPerf(monitor->Process(FrameDataInprocessing, &ResultItemSendToServer), "Total Tracking Process");
	// ��Ȧ�������Ŀ��
    // CheckPerf(searcher->SearchOneRound(FrameDataInprocessing), "Total cost while single round search");
	// ��Ȧ���������
	// CheckPerf(multiSearcher->SearchOneRound(FrameDataInprocessing, &ResultItemSendToServer), "Total cost while single round search");

	// �����洢�������������
	std::unique_lock<std::mutex> writerLock(resultBuffer->bufferMutex);
	while ((resultBuffer->write_position + 1) % RING_BUFFER_SIZE == resultBuffer->read_position)
	{
		// ע�ʹ�ӡ��־�����ⲻ��Ҫ��ʱ�俪��
		// printf("Result Send Producer is waiting for an empty slot...\n");
		resultBuffer->buffer_not_full.wait(writerLock);
	}

	// Copy data received from network to ring buffer and update ring buffer header pointer
	memcpy(resultBuffer->item_buffer + resultBuffer->write_position * ResultItemSize, &ResultItemSendToServer, ResultItemSize);
	resultBuffer->write_position++;

	// Reset data header pointer when to the end of buffer
	if (resultBuffer->write_position == RING_BUFFER_SIZE)
		resultBuffer->write_position = 0;

	// Notify Detect thread
	resultBuffer->buffer_not_empty.notify_all();
	writerLock.unlock();

	// ��ʱ��ʾ���
	auto shouLastResultDelay = 1;
	ShowLastResult(shouLastResultDelay);

	// ����һ���߳�ִ��״̬
	return true;
}

/****************************************************************************************/
/* �������壺����һ֡�����                                                             */
/****************************************************************************************/
bool OutputData(DetectResultRingBufferStruct* buffer)
{
	// ������ݽ��������
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while (buffer->write_position == buffer->read_position)
	{
		if (buffer->finish_flag == true)
			return false;

		//		printf("Result send thread is waiting for result items...\n");
		buffer->buffer_not_empty.wait(lock);
	}

	memcpy(&ResultItemSendToServer, buffer->item_buffer + buffer->read_position * ResultItemSize, ResultItemSize);
	SendResultToRemoteServer(ResultItemSendToServer);

	buffer->read_position++;

	if (buffer->read_position >= RING_BUFFER_SIZE)
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
		if (InputDataToBuffer(&Buffer) == false) break;

		timeBeginPeriod(1);
		Sleep(1);
		timeEndPeriod(1);
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
	detector->SetRemoveFalseAlarmParameters(false, false, false, false, true, true);

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

	// ɾ��ָ��
	RelaseSource();
}