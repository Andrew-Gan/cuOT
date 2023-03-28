#include <stdlib.h>
#include "mytypes.h"

bool isSent = false;
bool isReceived = false;
int currChoice = 0;
TreeNode currContent;

void ot_send(TreeNode& m0, TreeNode& m1) {
    while (!isReceived);
    if (currChoice == 0) {
        memcpy(&currContent, &m0, sizeof(m0));
    }
    else {
        memcpy(&currContent, &m1, sizeof(m1));
    }
    isSent = true;
}

TreeNode ot_recv(int choice) {
    currChoice = choice;
    isReceived = true;
    while (!isSent);
    return currContent;
}
