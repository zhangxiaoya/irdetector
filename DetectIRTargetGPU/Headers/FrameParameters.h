// ͼ����
#ifndef IMAGE_WIDTH
#define IMAGE_WIDTH (320 * 2)
#endif

// ͼ��߶�
#ifndef IMAGE_HEIGHT
#define IMAGE_HEIGHT (256 * 2)
#endif

// �������ش�С
#ifndef PIXEL_SIZE
#define PIXEL_SIZE (2)
#endif

// ֡���ݴ�С
#ifndef FRAME_DATA_SIZE
#define FRAME_DATA_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * PIXEL_SIZE)
#endif

// ͼ����������
#ifndef IMAGE_SIZE
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)
#endif

// �ֶδ���ͼ��ÿ�����ݴ�С
#ifndef SEGMENT_SIZE
#define SEGMENT_SIZE (320 * 64)
#endif

// ÿһ֡ͼ�����ݶεĸ���
#ifndef SEGMENT_COUNT
#define SEGMENT_COUNT (IMAGE_WIDTH * IMAGE_HEIGHT / SEGMENT_SIZE)
#endif
