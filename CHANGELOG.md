# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.0.0] - 2026-10-26

alpaka 3.0.0 is a complete rewrite with a new API and implementation. 
The user-facing interface relies less on complex templates, and new abstractions make it easier to write efficient code for both CPUs and GPUs, including SIMD vectorization. 
Explicit control over execution and memory remains central to the library.
The new API breaks compatibility with alpaka 2.x. Existing applications need to be ported.
