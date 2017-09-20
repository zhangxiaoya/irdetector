#pragma once

#include "../LogPrinter/LogPrinter.hpp"

extern LogPrinter logPrinter;

static auto ConvexPartitionOfOriginalImage = 0;
static auto ConcavePartitionOfOriginalImage = 0;

static auto ConvexPartitionOfDiscretizedImage = 0;
static auto ConcavePartitionOfDiscretizedImage = 0;

const auto MinDiffOfConvextAndConcaveThreshold = 3;