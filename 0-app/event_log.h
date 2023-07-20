#ifndef __EVENT_LOG_H__
#define __EVENT_LOG_H__

#include <fstream>
#include <mutex>

enum Event {
  CudaInit, BaseOT, Expand, Compress, Hash,

  // subevents of Expand
  ExpandInit, ExpandHash, ExpandSum, ExpandXor, ExpandSend, ExpandRecv,

  // subevents of Encode
  CompressInit, CompressTP, CompressFFT, CompressMult, CompressIFFT,

  NUM_EVENTS,
};

class Log {
private:
  static std::mutex mtx;
  static std::ofstream logFile[2];
  static struct timespec initTime;
  static float eventStart[2][NUM_EVENTS];
  static float eventDuration[2][NUM_EVENTS];

public:
  static void open(const char *filename, const char *filename2);
  static void close();
  static void start(int role, Event event);
  static void end(int role, Event event);
};

#endif
