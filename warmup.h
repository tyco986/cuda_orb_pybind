#pragma once


void warmup();

/* For determinism testing: run warmup with fixed seed and return checksum of result */
float warmup_capture_output(int seed);