
#include "Validation.h"
#include "CorrectnessValidation.hpp"
#include "PerformanceValidation.hpp"

/****************************************************************************************/
/*                          Test Algrithm Core Performance                              */
/****************************************************************************************/
void CheckConrrectness()
{
	auto validation_file_name = "D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin";

	CorrectnessValidation validator;
	validator.InitValidationData(validation_file_name);
	validator.VailidationAll();
}

void CheckPerformance()
{
//	auto validation_file_name = "D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin";
	//	auto validation_file_name = "D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1.bin";
	auto validation_file_name = "D:\\Cabins\\Projects\\Project1\\ir_file_20170925_220915_mubiaojingzhi.bin";

	PerformanceValidation validator;
	validator.InitDataReader(validation_file_name);
	validator.VailidationAll();
}