#include "DetectorParameters.h"

// ��תһȦͼ��֡��
#ifndef FRAME_COUNT_ONE_ROUND
#define FRAME_COUNT_ONE_ROUND (171)
#endif

// ��תһȦ������⵽��Ŀ������
#ifndef SEARCH_TARGET_COUNT_ONE_ROUND
#define SEARCH_TARGET_COUNT_ONE_ROUND (MAX_DETECTED_TARGET_COUNT * FRAME_COUNT_ONE_ROUND)
#endif

// ����ܳ���Ŀ���֡��
#ifndef FRAME_COUNT_WHICH_MOST_LIKELY_HAVE_TARGETS
#define FRAME_COUNT_WHICH_MOST_LIKELY_HAVE_TARGETS (3)
#endif

// ÿ���Ƕȸ��ٶ�����󳤶�
#ifndef MAX_TRACE_QUEUE_LENGTH
#define MAX_TRACE_QUEUE_LENGTH (20)
#endif // !MAX_TRACE_QUEUE_LENGTH
