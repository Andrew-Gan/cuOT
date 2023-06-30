#include <thread>
#include "event_log.h"

std::mutex Log::mtx;
std::ofstream Log::logFile[2];
struct timespec Log::initTime;

const char *eventString[] = {
  "BaseOT", "BufferInit", "PprfExpand", "SumNodes",
  "MatrixInit", "MatrixMult",
};

void Log::open(const char *filename, const char *filename2) {
  if (Log::logFile[0].is_open())
    Log::logFile[0].close();
  if (Log::logFile[1].is_open())
    Log::logFile[1].close();

  Log::logFile[0].open(filename, std::ofstream::out);
  Log::logFile[1].open(filename2, std::ofstream::out);

  for (int f = 0; f < 2; f++) {
    for (int i = 0; i < sizeof(eventString) / sizeof(eventString[0]); i++) {
      Log::logFile[f] << i << " " << eventString[i] << std::endl;
    }
    Log::logFile[f] << "--------------------" << std::endl;
    Log::logFile[f] << "<start/end> <event> <ms since init>" << std::endl;
    Log::logFile[f] << "--------------------" << std::endl;
  }
  clock_gettime(CLOCK_MONOTONIC, &Log::initTime);
}

void Log::close() {
  if (Log::logFile[0].is_open())
    Log::logFile[0].close();
  if (Log::logFile[1].is_open())
    Log::logFile[1].close();
}

void Log::start(int role, Event event) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - Log::initTime.tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - Log::initTime.tv_nsec) / 1000000.0;
  mtx.lock();
  Log::logFile[role] << "s " << event << " " << timeSinceStart << std::endl;
  mtx.unlock();
}

void Log::end(int role, Event event) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - Log::initTime.tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - Log::initTime.tv_nsec) / 1000000.0;
  mtx.lock();
  Log::logFile[role] << "e " << event << " " << timeSinceStart << std::endl;
  mtx.unlock();
}
