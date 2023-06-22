#pragma once


// use the miracl library for curves
// #define ENABLE_MIRACL @ENABLE_MIRACL@

// use the relic library for curves
// #define ENABLE_RELIC @ENABLE_RELIC@

// use the libsodium library for curves
#define ENABLE_SODIUM @ENABLE_SODIUM@

// does the libsodium library support noclamp operations on Montgomery curves?
// #define SODIUM_MONTGOMERY @SODIUM_MONTGOMERY@

// compile the circuit library
#define ENABLE_CIRCUITS @ENABLE_CIRCUITS@

// include the span-lite
// #define ENABLE_SPAN_LITE @ENABLE_SPAN_LITE@

// defined if we should use cpp 14 and undefined means cpp 11
#define ENABLE_CPP_14 @ENABLE_CPP_14@

// Turn on Channel logging for debugging.
// #define ENABLE_NET_LOG @ENABLE_NET_LOG@

// enable the wolf ssl socket layer.
// #define ENABLE_WOLFSSL @ENABLE_WOLFSSL@

// enable integration with boost for networking.
// #define ENABLE_BOOST @ENABLE_BOOST@

// enable the use of intel SSE instructions.
// #define ENABLE_SSE @ENABLE_SSE@

// enable the use of intel AVX instructions.
#define ENABLE_AVX @ENABLE_AVX@

// enable the use of the portable AES implementation.
// #define ENABLE_PORTABLE_AES @ENABLE_PORTABLE_AES@

#if (defined(_MSC_VER) || defined(__SSE2__)) && defined(ENABLE_SSE)
#define ENABLE_BLAKE2_SSE ON
#define OC_ENABLE_SSE2 ON
#endif

#if (defined(_MSC_VER) || defined(__PCLMUL__)) && defined(ENABLE_SSE)
#define OC_ENABLE_PCLMUL
#endif

#if (defined(_MSC_VER) || defined(__AES__)) && defined(ENABLE_SSE)
#define OC_ENABLE_AESNI ON
#else
#define OC_ENABLE_PORTABLE_AES ON
#endif

#if (defined(_MSC_VER) || defined(__AVX2__)) && defined(ENABLE_AVX)
#define OC_ENABLE_AVX2 ON
#endif
