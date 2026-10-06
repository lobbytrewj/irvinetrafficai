#!/usr/bin/env bash
#
# bootstrap_repo.sh — scaffold the irvine-traffic-ai repository layout.
#
# Idempotent: safe to re-run. Existing files are never overwritten.
#
# Usage:
#   ./scripts/bootstrap_repo.sh              # scaffold into repo root (parent of scripts/)
#   ./scripts/bootstrap_repo.sh /path/to/dir # scaffold into an explicit directory

set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$ROOT"
cd "$ROOT"

# ---------------------------------------------------------------------------
# Directory tree
# ---------------------------------------------------------------------------
DIRS=(
  data/raw
  data/processed
  cpp_engine/include
  cpp_engine/src
  src/data
  src/models
  src/pipelines
  src/router
  tests/python
  tests/cpp
  app
)

for d in "${DIRS[@]}"; do
  mkdir -p "$d"
  echo "  dir   $d"
done

# ---------------------------------------------------------------------------
# Marker files
#
# .gitkeep  -> keeps otherwise-empty data/test dirs in version control
# __init__.py -> makes the Python trees importable packages
# ---------------------------------------------------------------------------
write_if_absent() {
  local path="$1"
  local body="${2-}"
  if [[ -e "$path" ]]; then
    echo "  skip  $path (exists)"
    return
  fi
  printf '%s' "$body" > "$path"
  echo "  file  $path"
}

for d in data/raw data/processed cpp_engine/include cpp_engine/src tests/cpp; do
  write_if_absent "$d/.gitkeep" ""
done

for pkg in src src/data src/models src/pipelines src/router tests/python app; do
  write_if_absent "$pkg/__init__.py" ""
done

# ---------------------------------------------------------------------------
# Minimal C++ extension seed so `pip install -e .` builds on a fresh clone
# ---------------------------------------------------------------------------
write_if_absent cpp_engine/include/router.hpp '#pragma once

#include <cstdint>
#include <vector>

namespace irvine {

// Placeholder for the routing core. Replace with the real graph/CSR
// structures and shortest-path implementation.
struct RouteResult {
    std::vector<std::int64_t> path;
    double cost = 0.0;
};

// Returns the engine version string. Exists so the pybind11 module has
// something real to bind before the router lands.
const char* version() noexcept;

}  // namespace irvine
'

write_if_absent cpp_engine/src/router.cpp '#include "router.hpp"

namespace irvine {

const char* version() noexcept { return "0.1.0"; }

}  // namespace irvine
'

write_if_absent cpp_engine/src/bindings.cpp '#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "router.hpp"

namespace py = pybind11;

PYBIND11_MODULE(traffic_engine, m) {
    m.doc() = "Native routing engine for irvine-traffic-ai";

    m.def("version", &irvine::version, "Engine version string");

    py::class_<irvine::RouteResult>(m, "RouteResult")
        .def(py::init<>())
        .def_readwrite("path", &irvine::RouteResult::path)
        .def_readwrite("cost", &irvine::RouteResult::cost);
}
'

# ---------------------------------------------------------------------------
# Ignore generated + data artifacts
# ---------------------------------------------------------------------------
write_if_absent .gitignore '# Python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.pytest_cache/

# Native build output
build/
dist/
*.so
*.dylib
*.pyd

# Data (tracked dirs stay via .gitkeep, contents do not)
data/raw/*
data/processed/*
!data/**/.gitkeep

# Models
*.onnx
*.pt
*.pth
'

echo
echo "Scaffold complete under: $ROOT"
