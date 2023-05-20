#include <thread>
#include "event_log.h"

std::mutex EventLog::mtx;
std::ofstream EventLog::logFile;
struct timespec EventLog::initTime;

const char *eventString[] = {
  "AesKeyExpansion",
  "BufferInit", "MatrixInit", "MatrixRand",
  "BaseOTSend", "BaseOTRecv",
  "HashSender", "HashRecver",
  "PprfSenderExpand", "PprfRecverExpand",
};

void EventLog::open(const char *filename) {
  if (EventLog::logFile.is_open())
    EventLog::logFile.close();
  EventLog::logFile.open(filename, std::ofstream::out);
  for (int i = 0; i < sizeof(eventString) / sizeof(eventString[0]); i++) {
    EventLog::logFile << i << " " << eventString[i] << std::endl;
  }
  EventLog::logFile << "--------------------" << std::endl;
  EventLog::logFile << "<start/end> <thread id> <event> <ms since init>" << std::endl;
  EventLog::logFile << "--------------------" << std::endl;
  clock_gettime(CLOCK_MONOTONIC, &EventLog::initTime);
}

void EventLog::close() {
  if (EventLog::logFile.is_open())
    EventLog::logFile.close();
}

void EventLog::start(Event event) {
  std::thread::id tid = std::this_thread::get_id();
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - EventLog::initTime.tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - EventLog::initTime.tv_nsec) / 1000000.0;
  mtx.lock();
  EventLog::logFile << "s " << tid << " " << event << " " << timeSinceStart << std::endl;
  mtx.unlock();
}

void EventLog::end(Event event) {
  std::thread::id tid = std::this_thread::get_id();
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - EventLog::initTime.tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - EventLog::initTime.tv_nsec) / 1000000.0;
  mtx.lock();
  EventLog::logFile << "e " << tid << " " << event << " " << timeSinceStart << std::endl;
  mtx.unlock();
}
