#include <thread>
#include "event_log.h"

std::mutex Log::mtx;
std::ofstream Log::logFile[2];
struct timespec Log::initTime;
float Log::eventStart[2][NUM_EVENTS];
float Log::eventDuration[2][NUM_EVENTS];

const char *eventString[] = {
  "CudaInit", "BaseOT", "Expand", "Compress", "Hash",

  // subevents of Expand
  "ExpandInit", "ExpandHash", "ExpandSum", "ExpandXor", "ExpandSend", "ExpandRecv",

  // subevents of Encode
  "CompressInit", "CompressTP", "CompressFFT", "CompressMult", "CompressIFFT",
};

void Log::open(const char *filename, const char *filename2) {
  Log::logFile[0].open(filename, std::ofstream::out);
  Log::logFile[1].open(filename2, std::ofstream::out);

  for (int f = 0; f < 2; f++) {
    for (int i = 0; i < sizeof(eventString) / sizeof(eventString[0]); i++) {
      Log::logFile[f] << i << " " << eventString[i] << std::endl;
    }
    Log::logFile[f] << "--------------------" << std::endl;
  }
  clock_gettime(CLOCK_MONOTONIC, &Log::initTime);
}

void Log::close() {
  for (int role = 0; role < 2; role++) {
    for (int event = 0; event < NUM_EVENTS; event++) {
      Log::logFile[role] << "t " << event << " " << Log::eventDuration[role][event] << std::endl;
    }
  }
  Log::logFile[0].close();
  Log::logFile[1].close();
}

void Log::start(int role, Event event) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float timeSinceStart = (now.tv_sec - Log::initTime.tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - Log::initTime.tv_nsec) / 1000000.0;
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
  float timeSinceStart = (now.tv_sec - Log::initTime.tv_sec) * 1000;
  timeSinceStart += (now.tv_nsec - Log::initTime.tv_nsec) / 1000000.0;
  // only record start stop for main events
  if (event <= Hash) {
    mtx.lock();
    Log::logFile[role] << "e " << event << " " << timeSinceStart << std::endl;
    mtx.unlock();
  }
  Log::eventDuration[role][event] += timeSinceStart - Log::eventStart[role][event];
}
