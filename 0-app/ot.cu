#include "ot.h"

std::array<std::atomic<OT*>, 100> senders;
std::array<std::atomic<OT*>, 100> recvers;

OT::OT(Role myrole, int myid) : role(myrole), id(myid) {
  if (role == Sender)
    senders[id] = this;
  else
    recvers[id] = this;
}
