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

const bool IsSendResultToServer = false; // �Ƿ��ͽ���������

// ͼ����Ϣȫ�ֱ��������붨��
extern const unsigned int WIDTH = 320;   // ͼ����
extern const unsigned int HEIGHT = 256;  // ͼ��߶�
extern const unsigned int BYTESIZE = 2;  // ÿ�������ֽ���

static const int FrameSize = WIDTH * HEIGHT * BYTESIZE;        // ÿ��ͼ��֡�Ĵ�С
unsigned char FrameData[FrameSize];                            // ÿһ֡ͼ����ʱ����
unsigned char FrameDataInprocessing[FrameSize] = {0};          // ÿһ֡ͼ����ʱ����
ResultSegment ResultItem;                                      // ÿһ֡ͼ������
static const int ResultItemSize = sizeof(ResultSegment);       // ÿһ֡ͼ��������С

Detector* detector = new Detector();                  // ��ʼ�������

// ������ȫ�ֱ��������붨��
static const int BufferSize = 10;                     // �߳�ͬ����������С
RingBufferStruct Buffer(FrameSize, BufferSize);       // ���ݽ����̻߳��λ�������ʼ��
ResultBufferStruct ResultBuffer(BufferSize);          // ��������̻߳��λ�������ʼ��

/****************************************************************************************/
/*                                Input Data Operation                                  */
/****************************************************************************************/
bool InputDataToBuffer(RingBufferStruct* buffer)
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
	memcpy(buffer->item_buffer + buffer->write_position * FrameSize, FrameData, FrameSize * sizeof(unsigned char));
	buffer->write_position++;

	auto frameIndex = reinterpret_cast<int*>(FrameData + 2);
	std::cout<<" ====================================================================>" << *frameIndex << std::endl;

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

	memcpy(FrameDataInprocessing, buffer->item_buffer + buffer->read_position * FrameSize, FrameSize);

	buffer->read_position++;

	if (buffer->read_position >= BufferSize)
		buffer->read_position = 0;

	buffer->buffer_not_full.notify_all();
	readLock.unlock();

	// ���Ŀ�꣬���������
	CheckPerf(detector->DetectTargets(FrameDataInprocessing, &ResultItem), "Total process");

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
		memcpy(resultBuffer->item_buffer + resultBuffer->write_position * ResultItemSize, &ResultItem, ResultItemSize);
		resultBuffer->write_position++;

		// Reset data header pointer when to the end of buffer
		if (resultBuffer->write_position == BufferSize)
			resultBuffer->write_position = 0;

		// Notify Detect thread
		resultBuffer->buffer_not_empty.notify_all();
		writerLock.unlock();
	}

	// ����һ���߳�ִ��״̬
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

		std::cout << "Result send thread is waiting for result items...\n";
		buffer->buffer_not_empty.wait(lock);
	}

	memcpy(&ResultItem, buffer->item_buffer + buffer->read_position * ResultItemSize, ResultItemSize);
	SendResultToRemoteServer(ResultItem);
//	SendResultToRemoteServer(buffer->item_buffer[buffer->read_position]);

	buffer->read_position++;

	if (buffer->read_position >= BufferSize)
		buffer->read_position = 0;

	buffer->buffer_not_full.notify_all();
	lock.unlock();

	return true;
}

/****************************************************************************************/
/*                                  Input Data Task                                     */
/****************************************************************************************/
void InputDataTask()
{
	// ѭ���������ݣ��߳����ȼ�̫�ߣ�
	while (true)
	{
		if(InputDataToBuffer(&Buffer) == false) break;
		//Sleep(1);/////???????????????????????????????????????????
	}
}

/****************************************************************************************/
/*                                      Detect Task                                     */
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
/*                               Output Result Task                                     */
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
/*                               Initial Data Buffer                                    */
/****************************************************************************************/
void InitBuffer(RingBufferStruct* buffer)
{
	buffer->write_position = 0;
	buffer->read_position = 0;
}

/****************************************************************************************/
/*                               Initial Result Buffer                                  */
/****************************************************************************************/
void InitResultBuffer(ResultBufferStruct* buffer)
{
	buffer->read_position = 0;
	buffer->write_position = 0;
}

void RunOnNetwork()
{
	// ��ʼ��Socket���绷��
	InitNetworks();

	// ��ʼ������Ӿֲ��洢�ͼ�����
	detector->InitSpace();
	detector->SetAllParameters();

	// ��ʼ�����ݻ���ͽ������
	InitBuffer(&Buffer);
	InitResultBuffer(&ResultBuffer);

	// ���������̣߳���ȡ�����̡߳������������ؽ��
	std::thread InputDataThread(InputDataTask);
	std::thread DetectorThread(DetectTask);
//	std::thread OutputDataThread(OutputDataTask);

	// �����߳̿�ʼ����
	InputDataThread.join();
	DetectorThread.join();
//	OutputDataThread.join();

	// ��������
	DestroyNetWork();
}

void TestUsingBinaryFile()
{
	DetectorValidation visualEffectValidator;
	visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
//	visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1.bin");
	visualEffectValidator.VailidationAll();
}

void TestPerformance()
{
	/* ������CUDA�˺������Գ���͵����ı��ļ����ԣ������׶β�Ҫɾ����������� */
	Validation validation;
	validation.InitValidationData("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
	validation.VailidationAll();
}

/****************************************************************************************/
/*                                     Main Function                                    */
/****************************************************************************************/
int main(int argc, char* argv[])
{
	// ��ʼ��CUDA�豸
	auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
//		RunOnNetwork();

		TestPerformance();

//		TestUsingBinaryFile();
	}

	// ���ټ����
	delete detector;

	// �ͷ�CUDA�豸
	CUDAInit::cudaDeviceRelease();

	system("Pause");
	return 0;
}
