#ifndef __EVENT_LOG_H__
#define __EVENT_LOG_H__

#include <fstream>
#include <mutex>

enum Event {
  CudaInit, BaseOT, Expand, LPN, Neural, NUM_EVENTS,
};

const char *eventString[] = {
  "CudaInit", "BaseOT", "Expand", "Compress", "Hash",
};

class Log {
private:
  static std::mutex mtx;
  static std::ofstream logFile[2];
  static struct timespec initTime[2];
  static float eventStart[2][NUM_EVENTS];
  static float eventDuration[2][NUM_EVENTS];

public:
  static void open(int role, const char *filename) {
    Log::logFile[role].open(filename, std::ofstream::out);

    for (int i = 0; i < sizeof(eventString) / sizeof(eventString[0]); i++) {
      Log::logFile[role] << i << " " << eventString[i] << std::endl;
    }
    Log::logFile[role] << "--------------------" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &Log::initTime[role]);
  }

  static void close(int role) {
    for (int event = 0; event < NUM_EVENTS; event++) {
      Log::logFile[role] << "t " << event << " " << Log::eventDuration[role][event] << std::endl;
    }
    Log::logFile[role].close();
  }

  static void start(int role, Event event) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    float timeSinceStart = (now.tv_sec - Log::initTime[role].tv_sec) * 1000;
    timeSinceStart += (now.tv_nsec - Log::initTime[role].tv_nsec) / 1000000.0;
    mtx.lock();
    Log::logFile[role] << "s " << event << " " << timeSinceStart << std::endl;
    mtx.unlock();
    Log::eventStart[role][event] = timeSinceStart;
  }

  static void end(int role, Event event) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    float timeSinceStart = (now.tv_sec - Log::initTime[role].tv_sec) * 1000;
    timeSinceStart += (now.tv_nsec - Log::initTime[role].tv_nsec) / 1000000.0;
    mtx.lock();
    Log::logFile[role] << "e " << event << " " << timeSinceStart << std::endl;
    mtx.unlock();
    Log::eventDuration[role][event] += timeSinceStart - Log::eventStart[role][event];
  }
};

std::mutex Log::mtx;
std::ofstream Log::logFile[2];
struct timespec Log::initTime[2];
float Log::eventStart[2][NUM_EVENTS];
float Log::eventDuration[2][NUM_EVENTS];

#endif
