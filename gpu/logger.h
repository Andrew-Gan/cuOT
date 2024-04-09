#ifndef __EVENT_LOG_H__
#define __EVENT_LOG_H__

#include <fstream>
#include "gpu_define.h"

enum Event {
  CudaInit, BaseOT, SeedExp, LPN, Neural, NUM_EVENTS,
};

class Log {
private:
  static std::ofstream logFile[2];
  static struct timespec initTime[2];
  static float eventStart[2][NUM_EVENTS];
  static float eventDuration[2][NUM_EVENTS];
  static uint64_t memStart[2][NUM_EVENTS];
  static uint64_t memCurr[2][NUM_EVENTS];
  static uint64_t memMax[2][NUM_EVENTS];
  static bool mOpened[2], mIgnoreInit[2], initTimeSet[2];

public:
  static void open(Role role, std::string filename, bool ignoreInit);
  static void close(Role role);
  static void start(Role role, Event event);
  static void end(Role role, Event event);
  static void mem(Role role, Event event);
};

#endif
