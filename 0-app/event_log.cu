#include <thread>
#include "event_log.h"

std::mutex EventLog::mtx;
std::ofstream EventLog::logFile[2];
struct timespec EventLog::initTime;

const char *eventString[] = {
  "BufferInit", "BaseOT", "PprfExpand", "SumNodes", "Hash",
  "MatrixInit", "MatrixRand", "MatrixMult",
};

void EventLog::open(const char *filename, const char *filename2) {
  if (EventLog::logFile[0].is_open())
    EventLog::logFile[0].close();
  if (EventLog::logFile[1].is_open())
    EventLog::logFile[1].close();

  EventLog::logFile[0].open(filename, std::ofstream::out);
  EventLog::logFile[1].open(filename2, std::ofstream::out);

  for (int f = 0; f < 2; f++) {
    for (int i = 0; i < sizeof(eventString) / sizeof(eventString[0]); i++) {
      EventLog::logFile[f] << i << " " << eventString[i] << std::endl;
    }
    EventLog::logFile[f] << "--------------------" << std::endl;
    EventLog::logFile[f] << "<start/end> <event> <ms since init>" << std::endl;
    EventLog::logFile[f] << "--------------------" << std::endl;
  }
  clock_gettime(CLOCK_MONOTONIC, &EventLog::initTime);
}

void EventLog::close() {
  if (EventLog::logFile[0].is_open())
    EventLog::logFile[0].close();
  if (EventLog::logFile[1].is_open())
    EventLog::logFile[1].close();
}

void EventLog::start(int role, Event event) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - EventLog::initTime.tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - EventLog::initTime.tv_nsec) / 1000000.0;
  mtx.lock();
  EventLog::logFile[role] << "s " << event << " " << timeSinceStart << std::endl;
  mtx.unlock();
}

void EventLog::end(int role, Event event) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - EventLog::initTime.tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - EventLog::initTime.tv_nsec) / 1000000.0;
  mtx.lock();
  EventLog::logFile[role] << "e " << event << " " << timeSinceStart << std::endl;
  mtx.unlock();
}
