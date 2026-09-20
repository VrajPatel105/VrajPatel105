# Vraj Patel

CS senior at UMass Dartmouth (May 2027). I work on ML systems and GPU inference: writing kernels in Triton and CUDA, building serving infrastructure from scratch, and benchmarking everything against the production library it replaces.

Looking for new-grad roles in ML systems, inference, and GPU performance starting 2027.

## Projects

**[paged-inference-engine](https://github.com/VrajPatel105/paged-inference-engine)**
A mini vLLM-style inference server built from scratch: paged KV cache, a continuous batching scheduler, and a FlashAttention-2 Triton kernel rewritten to read through a block table, with INT8 KV cache quantization. The paged kernel matches PyTorch SDPA within fp16 precision on scrambled, non-contiguous block layouts.

**[cpp-gpu-inference](https://github.com/VrajPatel105/cpp-gpu-inference)**
A GPU inference stack from first principles. FlashAttention-2 forward and backward in Triton (peak memory matches PyTorch SDPA up to 8K tokens), LLM.int8()-style quantization written from scratch (perplexity within 0.02% of the unquantized model), a Triton matmul within ~20% of cuBLAS in true FP32, and CUDA kernels profiled with Nsight Compute. [Full writeup](https://vrajpatel105.github.io/gpu-inference/index.html)

**[paged-engine-serving](https://github.com/VrajPatel105/paged-engine-serving)**
Serving layer for the paged engine: FastAPI, Docker, GitHub Actions CI/CD pushing to AWS ECR, and a health check that gates on Triton kernel warmup.

**[NutriGrove](https://github.com/VrajPatel105/NutriGrove)**
Mobile app recommending meals from live dining hall menus. Shipped on the App Store and Google Play with 100+ users; 2nd place at a $12,000 AI + startup hackathon.

## Experience

**Fleet Robotics**, ML Engineer Intern (Summer 2026). Closed the sim-to-real gap for a hull-cleaning robot by finding that the MuJoCo simulator's noise averaged out instead of accumulating, rebuilding the noise model against AprilTag ground truth, and retraining the SAC navigation policy on the corrected simulator.

**Blue Tech Externship**, Courage Builder Program (Mar 2026). Built a simulated autonomous UUV for sonar-based mine detection in MATLAB/Simulink; 1st place in the cohort.

**SMAST, UMass Dartmouth**, Research Software Engineering Intern (Fall 2025). Refactored the R codebase for a NOAA-funded offshore wind and fishery survey project, parallelized spatial simulations for a 2x speedup, and packaged an internal R library.

**SMAST, UMass Dartmouth**, Data Science Intern (Summer 2025). Automated analysis of 38 years of estuary data across 7 sites, cutting report turnaround from 3 weeks to 10 minutes.

## Stack

Python, C++, CUDA, Triton, PyTorch, Nsight Compute/Systems, FastAPI, Docker, AWS, Snowflake

## Links

[Resume](https://vrajpatel105.github.io/resume.pdf) · [Portfolio](https://vrajpatel105.github.io/) · [LinkedIn](https://linkedin.com/in/vrajpatel105) · vrajpatel.jobs@gmail.com
