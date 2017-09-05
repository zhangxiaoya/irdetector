#ifndef __GLOBAL_MAIN_HEADER__
#define __GLOBAL_MAIN_HEADER__

#include "cuda_runtime.h"
#include "driver_types.h"
#include "../LogPrinter/LogPrinter.hpp"
#include "../Models/LogLevel.hpp"

LogPrinter logPrinter;
const unsigned int WIDTH = 320;
const unsigned int HEIGHT = 256;
const unsigned BYTESIZE = 2;

#endif __GLOBAL_MAIN_HEADER__