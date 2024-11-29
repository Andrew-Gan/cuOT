#include <cuda.h>
#include "logger.h"

const char *eventString[] = {
  "CudaInit", "BaseOT", "SeedExp", "LPN", "Neural",
};

std::ofstream Log::logFile[2];
struct timespec Log::initTime[2];
float Log::eventStart[2][NUM_EVENTS];
float Log::eventDuration[2][NUM_EVENTS];
bool Log::mOpened[2] = {false, false};
int Log::mSampleSize = 1;

void Log::open(Role role, std::string filename, int sampleSize) {
  mSampleSize = sampleSize;
  mOpened[role] = true;
  logFile[role].open(filename);
  for (int i = 0; i < NUM_EVENTS; i++) {
    logFile[role] << i << " " << eventString[i] << std::endl;
  }
  logFile[role] << "--------------------" << std::endl;

  clock_gettime(CLOCK_MONOTONIC, &initTime[role]);
}

void Log::close(Role role) {
  mOpened[role] = false;
  for (int event = 0; event < NUM_EVENTS; event++)
    logFile[role] << "t " << event << " " << eventDuration[role][event] / mSampleSize << std::endl;
  logFile[role].close();
}

void Log::start(Role role, Event event) {
  if (!mOpened[role]) return;
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - initTime[role].tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - initTime[role].tv_nsec) / 1000000.0;
  logFile[role] << "s " << event << " " << timeSinceStart << std::endl;
  eventStart[role][event] = timeSinceStart;
}

void Log::end(Role role, Event event) {
  if (!mOpened[role]) return;
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - initTime[role].tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - initTime[role].tv_nsec) / 1000000.0;
  logFile[role] << "e " << event << " " << timeSinceStart << std::endl;
  eventDuration[role][event] += timeSinceStart - eventStart[role][event];
}

