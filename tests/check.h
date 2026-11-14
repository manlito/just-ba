#pragma once

// Minimal test helpers: no framework dependency. A test is an executable that
// returns 0 on success; CTest runs it. Return 77 to report "skipped".

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace jba_test {
inline int failures = 0;
}

#define EXPECT(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "%s:%d: EXPECT failed: %s\n", __FILE__, __LINE__, #cond); \
      ++jba_test::failures;                                                      \
    }                                                                            \
  } while (0)

#define EXPECT_NEAR(a, b, tol)                                                    \
  do {                                                                           \
    const double _a = (a), _b = (b), _t = (tol);                                 \
    if (!(std::abs(_a - _b) <= _t)) {                                            \
      std::fprintf(stderr, "%s:%d: EXPECT_NEAR failed: %s = %.17g, %s = %.17g, tol %.3g\n", \
                   __FILE__, __LINE__, #a, _a, #b, _b, _t);                      \
      ++jba_test::failures;                                                      \
    }                                                                            \
  } while (0)

#define EXPECT_THROWS(expr)                                                       \
  do {                                                                           \
    bool _threw = false;                                                         \
    try {                                                                        \
      (void)(expr);                                                              \
    } catch (...) {                                                              \
      _threw = true;                                                             \
    }                                                                            \
    if (!_threw) {                                                               \
      std::fprintf(stderr, "%s:%d: expected exception from %s\n", __FILE__, __LINE__, #expr); \
      ++jba_test::failures;                                                      \
    }                                                                            \
  } while (0)

#define TEST_MAIN_RESULT()                                                       \
  (jba_test::failures == 0 ? (std::printf("ok\n"), 0)                            \
                           : (std::fprintf(stderr, "%d failure(s)\n", jba_test::failures), 1))
