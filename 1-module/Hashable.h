#ifndef __HASHABLE_H__
#define __HASHABLE_H__

// This file and the associated implementation has been placed in the public domain, waiving all copyright. No restrictions are placed on its use.
#include <type_traits>
#include <cstring>

// Specializations of Hashable should inherit from std::true_type and contain:
//
// template<typename Hasher>
// static void hash(const T& t, Hasher& mHasher);
//
// Hasher will contain an Update method that can be applied to byte arrays and to Hashable
// types.
template<typename T, typename Enable = void>
struct Hashable : std::false_type {};

template<typename T>
struct Hashable<T, typename std::enable_if<std::is_pod<T>::value>::type> : std::true_type
{
    template<typename Hasher>
    static void hash(const T& t, Hasher& hasher)
    {
        hasher.Update((uint8_t*) &t, sizeof(T));
    }
};

#endif
