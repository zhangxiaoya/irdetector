
#include "Validation.h"
#include "CorrectnessValidation.hpp"
#include "PerformanceValidation.hpp"

/****************************************************************************************/
/*                          Test Algrithm Core Performance                              */
/****************************************************************************************/
void CheckConrrectness(const int width, const int height)
{
	const auto validation_file_name = "D:\\Cabins\\Projects\\Project1\\2\\ir_file_20170531_1000m_1_partOne.bin";

	CorrectnessValidation validator(width, height);
	validator.InitValidationData(validation_file_name);
	validator.VailidationAll();
}

void CheckPerformance(const int width, const int height, const int dilationRadius)
{
//	const auto validation_file_name = "D:\\Cabins\\Projects\\Project1\\2\\ir_file_20170531_1000m_1_partOne.bin";
//	const auto validation_file_name = "D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1.bin";
	const auto validation_file_name = "D:\\Cabins\\Projects\\Project1\\8\\ir_file_20170925_220915_mubiaojingzhi.bin";
//	const auto validation_file_name = "D:\\Cabins\\Projects\\Project1\\4\\test\\ir_file_20170713_300m_jingzhi.bin";

	PerformanceValidation validator(width, height, dilationRadius);
	validator.InitDataReader(validation_file_name);
	validator.VailidationAll();
}