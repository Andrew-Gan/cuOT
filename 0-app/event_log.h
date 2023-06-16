#ifndef __LOGGER_H__
#define __LOGGER_H__

#include <fstream>
#include <mutex>

enum Event {
  AesKeyExpansion, BufferInit,
  PprfSenderExpand, PprfRecverExpand, SumNodes,
  BaseOTSend, BaseOTRecv,
  MatrixInit, MatrixRand,
  HashSender, HashRecver,
};

class EventLog {
private:
  static std::mutex mtx;
  static std::ofstream logFile;
  static struct timespec initTime;

public:
  static void open(const char *filename);
  static void close();
  static void start(Event event);
  static void end(Event event);
};

#endif
