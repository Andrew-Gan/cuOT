#include <thread>
#include "event_log.h"

std::mutex Log::mtx;
std::ofstream Log::logFile[2];
struct timespec Log::initTime[2];
float Log::eventStart[2][NUM_EVENTS];
float Log::eventDuration[2][NUM_EVENTS];

const char *eventString[] = {
  "CudaInit", "BaseOT", "Expand", "Compress", "Hash",
};

void Log::open(int role, const char *filename) {
  Log::logFile[role].open(filename, std::ofstream::out);

  for (int i = 0; i < sizeof(eventString) / sizeof(eventString[0]); i++) {
    Log::logFile[role] << i << " " << eventString[i] << std::endl;
  }
  Log::logFile[role] << "--------------------" << std::endl;
  clock_gettime(CLOCK_MONOTONIC, &Log::initTime[role]);
}

void Log::close(int role) {
  for (int event = 0; event < NUM_EVENTS; event++) {
    Log::logFile[role] << "t " << event << " " << Log::eventDuration[role][event] << std::endl;
  }
  Log::logFile[role].close();
}

void Log::start(int role, Event event) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - Log::initTime[role].tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - Log::initTime[role].tv_nsec) / 1000000.0;
  // only record start stop for main events
  if (event <= Hash) {
    mtx.lock();
    Log::logFile[role] << "s " << event << " " << timeSinceStart << std::endl;
    mtx.unlock();
  }
  Log::eventStart[role][event] = timeSinceStart;
}

void Log::end(int role, Event event) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - Log::initTime[role].tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - Log::initTime[role].tv_nsec) / 1000000.0;
  // only record start stop for main events
  if (event <= Hash) {
    mtx.lock();
    Log::logFile[role] << "e " << event << " " << timeSinceStart << std::endl;
    mtx.unlock();
  }
  Log::eventDuration[role][event] += timeSinceStart - Log::eventStart[role][event];
}
