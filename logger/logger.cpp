#include "logger.h"

const char *eventString[] = {
  "CudaInit", "BaseOT", "SeedExp", "LPN", "Neural",
};

std::mutex Log::mtx;
std::ofstream Log::logFile[2];
struct timespec Log::initTime[2];
float Log::eventStart[2][NUM_EVENTS];
float Log::eventDuration[2][NUM_EVENTS];
uint64_t Log::commBytes[NUM_EVENTS] = {0};
uint64_t Log::bandwidth_mbps = 1;

void Log::open(int role, const char *filename, uint64_t mbps) {
  Log::logFile[role].open(filename, std::ofstream::out);
  Log::bandwidth_mbps = mbps;
  for (int i = 0; i < NUM_EVENTS; i++) {
    Log::logFile[role] << i << " " << eventString[i] << std::endl;
  }
  Log::logFile[role] << "--------------------" << std::endl;
  clock_gettime(CLOCK_MONOTONIC, &Log::initTime[role]);
}

void Log::close(int role) {
  for (int event = 0; event < NUM_EVENTS; event++) {
    Log::logFile[role] << "t " << event << " " << Log::eventDuration[role][event] << std::endl;
  }
  for (int i = 0; i < NUM_EVENTS; i++) {
    float megabytes = (float)commBytes[i] / 1000000;
    float seconds = megabytes / bandwidth_mbps;
    float miliseconds = seconds * 1000;
    Log::logFile[role] << "c " << i << " " << miliseconds << std::endl;
  }
  Log::logFile[role].close();
}

void Log::start(int role, Event event) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - Log::initTime[role].tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - Log::initTime[role].tv_nsec) / 1000000.0;
  mtx.lock();
  Log::logFile[role] << "s " << event << " " << timeSinceStart << std::endl;
  mtx.unlock();
  Log::eventStart[role][event] = timeSinceStart;
}

void Log::end(int role, Event event) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - Log::initTime[role].tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - Log::initTime[role].tv_nsec) / 1000000.0;
  mtx.lock();
  Log::logFile[role] << "e " << event << " " << timeSinceStart << std::endl;
  mtx.unlock();
  Log::eventDuration[role][event] += timeSinceStart - Log::eventStart[role][event];
}

void Log::comm(Event event, uint64_t bytes) {
  mtx.lock();
  commBytes[event] += bytes;
  mtx.unlock();
}
