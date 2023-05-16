#include "ot.h"

OT::OT(Role myrole, int myid) : role(myrole), id(myid) {
  if (role == Sender)
    senders[id] = this;
  else
    recvers[id] = this;
}
