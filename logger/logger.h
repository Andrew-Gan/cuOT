#ifndef __EVENT_LOG_H__
#define __EVENT_LOG_H__

#include <fstream>
#include <mutex>

enum Event {
  CudaInit, BaseOT, SeedExp, LPN, Neural, NUM_EVENTS,
};

class Log {
private:
  static std::mutex mtx;
  static std::ofstream logFile[2];
  static struct timespec initTime[2];
  static float eventStart[2][NUM_EVENTS];
  static float eventDuration[2][NUM_EVENTS];
  static uint64_t commBytes[NUM_EVENTS];
  static uint64_t bandwidth_mbps;

public:
  static void open(int role, const char *filename, uint64_t mbps);
  static void close(int role);
  static void start(int role, Event event);
  static void end(int role, Event event);
  static void comm(Event event, uint64_t bytes);
};

#endif
