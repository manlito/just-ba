// WebAssembly entry points (exploration): solve a problem file in Emscripten's
// in-memory file system on the browser's WebGPU device or the CPU, and write the
// synthetic problems of the benchmark (tests/make_problem.cpp, compiled in with its
// main renamed). Called from JavaScript through ccall; jba_solve suspends on WebGPU
// (JSPI), so it is called with {async: true}.

#include <emscripten/emscripten.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include "jba/problem_io.h"
#include "jba/solver.h"

int jba_make_problem_main(int argc, char** argv);  // tests/make_problem.cpp

namespace {

std::string result;  // the last answer; valid until the next call

bool ends_with(const std::string& s, const std::string& end) {
  return s.size() >= end.size() && s.compare(s.size() - end.size(), end.size(), end) == 0;
}

std::string escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if (c == '\n') {
      o += "\\n";
      continue;
    }
    o += c;
  }
  return o;
}

std::string number(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.12g", v);
  return buf;
}

}  // namespace

extern "C" {

// "webgpu" or "cpu"; whether the backend can run here.
EMSCRIPTEN_KEEPALIVE int jba_backend_available(const char* backend) {
  const auto b = jba::backend_from_name(backend);
  return b && jba::backend_available(*b) ? 1 : 0;
}

// Writes a synthetic problem to `path` from jba_make_problem's options, one string
// ("--model PINHOLE --images 500 ..."). Returns an empty string or the error.
EMSCRIPTEN_KEEPALIVE const char* jba_generate(const char* path, const char* options) {
  std::vector<std::string> args{"jba_make_problem", path};
  std::istringstream in(options);
  for (std::string a; in >> a;) args.push_back(a);
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  result.clear();
  if (jba_make_problem_main(static_cast<int>(argv.size()), argv.data()) != 0) result = "generation failed";
  return result.c_str();
}

// Solves the problem at `path` (.jba, or BAL text for .txt) and returns a JSON summary
// with the iteration history. warm_up: solve once untimed first (the device and its
// kernels are then ready, as the CLI's --warm-up).
EMSCRIPTEN_KEEPALIVE const char* jba_solve(const char* path, const char* backend, const char* linear_solver,
                                           int max_iterations, int warm_up) {
  try {
    const std::string p = path;
    const auto t0 = std::chrono::steady_clock::now();
    const jba::Problem start = ends_with(p, ".txt") ? jba::read_bal(p) : jba::read_problem(p);
    const double read_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    jba::SolverOptions o = jba::with_problem_losses({}, start);
    const auto b = jba::backend_from_name(backend);
    const auto l = jba::linear_solver_from_name(linear_solver);
    if (!b) throw std::invalid_argument(std::string("unknown backend ") + backend);
    if (!l) throw std::invalid_argument(std::string("unknown linear solver ") + linear_solver);
    o.backend = *b;
    o.linear_solver = *l;
    o.num_threads = 1;  // no threads in this build
    if (max_iterations > 0) o.max_iterations = max_iterations;
    if (warm_up) {
      jba::Problem w = start;
      jba::solve(w, o);
    }
    jba::Problem problem = start;
    const jba::SolverSummary s = jba::solve(problem, o);
    std::ostringstream j;
    j << "{\"ok\": true, \"backend\": \"" << backend << "\", \"linear_solver\": \""
      << jba::linear_solver_name(s.linear_solver) << "\", \"images\": " << start.num_images()
      << ", \"points\": " << start.num_points() << ", \"observations\": " << start.num_observations()
      << ", \"initial_cost\": " << number(s.initial_cost) << ", \"final_cost\": " << number(s.final_cost)
      << ", \"iterations\": " << s.iterations << ", \"linear_iterations\": " << s.linear_iterations
      << ", \"termination\": \"" << jba::termination_name(s.termination) << "\", \"message\": \""
      << escape(s.message) << "\", \"time_seconds\": " << number(s.time_seconds)
      << ", \"setup_seconds\": " << number(s.setup_seconds) << ", \"read_seconds\": " << number(read_seconds)
      << ", \"history\": [";
    for (std::size_t i = 0; i < s.history.size(); ++i)
      j << (i ? ", " : "") << "[" << number(s.history[i].cost) << ", " << number(s.history[i].time_seconds) << ", "
        << s.history[i].linear_iterations << "]";
    j << "]}";
    result = j.str();
  } catch (const std::exception& e) {
    result = std::string("{\"ok\": false, \"error\": \"") + escape(e.what()) + "\"}";
  }
  return result.c_str();
}

}  // extern "C"
