#pragma once
#include <string>
#include "../Models/LogLevel.hpp"
#include <iostream>
#include <ctime>
#include <sstream>

class LogPrinter
{
public:
	static void PrintLogs(std::string text, LogLevel logLevel);

	static void PrintLogs(std::string frontText, LogLevel logLevel, int value);

private:
	static void printCurrentTime();
};

inline void LogPrinter::PrintLogs(std::string text, LogLevel logLevel)
{
	printCurrentTime();

	switch (logLevel)
	{
	case LogLevel::Error:
		{
			std::cout << "ERROR => ";
			break;
		}
	case LogLevel::Info:
		{
			std::cout << "INFO => ";
			break;
		}
	case LogLevel::Waring:
		{
			std::cout << "WARNING => ";
			break;
		}
	default:
		{
			std::cout << "Not Defined => ";
			break;
		};
	}

	std::cout << text << std::endl;
}

inline void LogPrinter::PrintLogs(std::string frontText, LogLevel logLevel, int value)
{
	std::ostringstream textStream;
	textStream << frontText << value;
	PrintLogs(textStream.str(), logLevel);
}

inline void LogPrinter::printCurrentTime()
{
	const auto bufferSize = 20;

	auto timer = time(nullptr);
	auto currentTime = localtime(&timer);

	char timeStr[bufferSize];
	sprintf_s(timeStr, bufferSize, "%d-%02d-%02d %02d:%02d:%02d",
	          currentTime->tm_year + 1900,
	          currentTime->tm_mon + 1,
	          currentTime->tm_mday,
	          currentTime->tm_hour,
	          currentTime->tm_min,
	          currentTime->tm_sec);
	std::cout << timeStr << " ";
}

