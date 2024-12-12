#pragma once


#include <cassert>
#include <iostream>


#define TEST_ALL_TYPES(function_name, ...)                   \
   try {                                                     \
      function_name<int>(__VA_OPT__(__VA_ARGS__));           \
      function_name<long int>(__VA_ARGS__);                  \
      function_name<float>(__VA_ARGS__);                     \
      function_name<double>(__VA_ARGS__);                    \
      std::cout << "passed " << #function_name << std::endl; \
   } catch(...) {                                            \
      assert(false);                                         \
   }


#define TEST_ALL_FLOAT_TYPES(function_name, ...)             \
   try {                                                     \
      function_name<float>(__VA_ARGS__);                     \
      function_name<double>(__VA_ARGS__);                    \
      std::cout << "passed " << #function_name << std::endl; \
   } catch(...) {                                            \
      assert(false);                                         \
   }


#define TEST(function_name, ...)                             \
   try {                                                     \
      function_name(__VA_ARGS__);                            \
      std::cout << "passed " << #function_name << std::endl; \
   } catch(...) {                                            \
      assert(false);                                         \
   }

