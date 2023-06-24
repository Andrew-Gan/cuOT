
#ifndef sodium_version_H
#define sodium_version_H

#include "export.h"

#define SODIUM_VERSION_STRING "@VERSION@"

#define SODIUM_LIBRARY_VERSION_MAJOR 1
#define SODIUM_LIBRARY_VERSION_MINOR 0

#ifdef __cplusplus
extern "C" {
#endif

const char *sodium_version_string(void);

int         sodium_library_version_major(void);

int         sodium_library_version_minor(void);

int         sodium_library_minimal(void);

#ifdef __cplusplus
}
#endif

#endif
