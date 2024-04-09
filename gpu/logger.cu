#include <cuda.h>
#include "logger.h"

const char *eventString[] = {
  "CudaInit", "BaseOT", "SeedExp", "LPN", "Neural",
};

std::ofstream Log::logFile[2];
struct timespec Log::initTime[2];
float Log::eventStart[2][NUM_EVENTS];
float Log::eventDuration[2][NUM_EVENTS];
uint64_t Log::memStart[2][NUM_EVENTS] = {0};
uint64_t Log::memCurr[2][NUM_EVENTS] = {0};
uint64_t Log::memMax[2][NUM_EVENTS] = {0};
bool Log::mOpened[2] = {false, false};
bool Log::mIgnoreInit[2] = {false, false};
bool Log::initTimeSet[2] = {false, false};

void Log::open(Role role, std::string filename, bool ignoreInit) {
  mOpened[role] = true;
  mIgnoreInit[role] = ignoreInit;
  logFile[role].open(filename);
  for (int i = 0; i < NUM_EVENTS; i++) {
    logFile[role] << i << " " << eventString[i] << std::endl;
  }
  logFile[role] << "--------------------" << std::endl;


  if (!mIgnoreInit[role]) {
    clock_gettime(CLOCK_MONOTONIC, &initTime[role]);
    initTimeSet[role] = true;
  }

  memset(memCurr, 0, sizeof(memCurr));
  memset(memMax, 0, sizeof(memMax));
}

void Log::close(Role role) {
  mOpened[role] = false;
  for (int event = 0; event < NUM_EVENTS; event++) {
    logFile[role] << "t " << event << " " << eventDuration[role][event] << std::endl;
  }
  for (int j = 0; j < NUM_EVENTS; j++)
    logFile[role] << "m " << j << " " << memMax[role][j] << std::endl;
  logFile[role].close();
}

void Log::start(Role role, Event event) {
  if (!mOpened[role]) return;
  if (!initTimeSet[role] && mIgnoreInit[role] && event > CudaInit) {
    clock_gettime(CLOCK_MONOTONIC, &initTime[role]);
    initTimeSet[role] = true;
  }

  if (!mIgnoreInit[role] || event != CudaInit) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    float timeSinceStart = (now.tv_sec - initTime[role].tv_sec) * 1000;
    timeSinceStart += (now.tv_nsec - initTime[role].tv_nsec) / 1000000.0;
    logFile[role] << "s " << event << " " << timeSinceStart << std::endl;
    eventStart[role][event] = timeSinceStart;
  }

  size_t free, total, used;
	cudaMemGetInfo(&free, &total);
  used = total - free;
  memStart[role][event] = used;
  memCurr[role][event] = 0;
}

void Log::end(Role role, Event event) {
  if (!mOpened[role]) return;
  if (!mIgnoreInit[role] || event != CudaInit) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    float timeSinceStart = (now.tv_sec - initTime[role].tv_sec) * 1000;
    timeSinceStart += (now.tv_nsec - initTime[role].tv_nsec) / 1000000.0;
    logFile[role] << "e " << event << " " << timeSinceStart << std::endl;
    eventDuration[role][event] += timeSinceStart - eventStart[role][event];
  }

  size_t currMax = memCurr[role][event] - memStart[role][event];
  memMax[role][event] = std::max(currMax, memMax[role][event]);
}

void Log::mem(Role role, Event event) {
  if (!mOpened[role]) return;
  size_t free, total, used;
	cudaMemGetInfo(&free, &total);
  used = total - free;
  memCurr[role][event] = std::max(used, memCurr[role][event]);
}
