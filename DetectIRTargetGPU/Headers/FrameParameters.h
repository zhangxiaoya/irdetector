// 图像宽度
#ifndef IMAGE_WIDTH
#define IMAGE_WIDTH (320 * 2)
#endif

// 图像高度
#ifndef IMAGE_HEIGHT
#define IMAGE_HEIGHT (256 * 2)
#endif

// 单个像素大小
#ifndef PIXEL_SIZE
#define PIXEL_SIZE (2)
#endif

// 帧数据大小
#ifndef FRAME_DATA_SIZE
#define FRAME_DATA_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * PIXEL_SIZE)
#endif

// 图像像素数量
#ifndef IMAGE_SIZE
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)
#endif

// 分段传送图像，每段数据大小
#ifndef SEGMENT_SIZE
#define SEGMENT_SIZE (320 * 64)
#endif

// 每一帧图像数据段的个数
#ifndef SEGMENT_COUNT
#define SEGMENT_COUNT (IMAGE_WIDTH * IMAGE_HEIGHT / SEGMENT_SIZE)
#endif
