#ifndef __EVENT_LOG_H__
#define __EVENT_LOG_H__

#include <fstream>
#include <mutex>

enum Event {
  BaseOT, BufferInit, PprfExpand, SumNodes,
  MatrixInit, MatrixMult,
};

class Log {
private:
  static std::mutex mtx;
  static std::ofstream logFile[2];
  static struct timespec initTime;

public:
  static void open(const char *filename, const char *filename2);
  static void close();
  static void start(int role, Event event);
  static void end(int role, Event event);
};

#endif
