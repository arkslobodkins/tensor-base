#pragma once

#include <chrono>
#include <cstdio>

#define TIME_EXPR(a)                                               \
   do {                                                            \
      tnb::timer t;                                                \
      a;                                                           \
      std::printf("%s took: %.4e seconds\n", (#a), t.wall_time()); \
   } while(0)


namespace tnb {


struct timer {
   explicit timer() {
      start = std::chrono::high_resolution_clock::now();
   }

   void restart() {
      start = std::chrono::high_resolution_clock::now();
   }

   double wall_time() {
      return double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now() - start)
                        .count())
           / 1.e9;
   }

private:
   std::chrono::system_clock::time_point start;
};


}  // namespace tnb

