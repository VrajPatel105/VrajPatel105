# Vraj Patel

CS senior at UMass Dartmouth (May 2027). I work on ML systems and GPU inference: writing kernels in Triton and CUDA, building serving infrastructure from scratch, and benchmarking everything against the production library it replaces.

Looking for new-grad roles in ML systems, inference, and GPU performance starting 2027.

## Projects

**[paged-inference-engine](https://github.com/VrajPatel105/paged-inference-engine)**
A mini vLLM-style inference server built from scratch: paged KV cache, a continuous batching scheduler, and a FlashAttention-2 Triton kernel rewritten to read through a block table, with INT8 KV cache quantization. The paged kernel matches PyTorch SDPA within fp16 precision on scrambled, non-contiguous block layouts.

**[blackwell-gemm](https://github.com/VrajPatel105/blackwell-gemm)**
FP32 SGEMM in CUDA C++: 7 kernels from naive to warp tiling, each profiled with Nsight Compute. Best: 76% of cuBLAS on an RTX 5080 Laptop, 56% on a B200.

**[cpp-gpu-inference](https://github.com/VrajPatel105/cpp-gpu-inference)**
A GPU inference stack from first principles. FlashAttention-2 forward and backward in Triton (peak memory matches PyTorch SDPA up to 8K tokens), LLM.int8()-style quantization written from scratch (perplexity within 0.02% of the unquantized model), a Triton matmul within ~20% of cuBLAS in true FP32, and CUDA kernels profiled with Nsight Compute. [Full writeup](https://vrajpatel105.github.io/gpu-inference/index.html)

**[paged-engine-serving](https://github.com/VrajPatel105/paged-engine-serving)**
Serving layer for the paged engine: FastAPI, Docker, GitHub Actions CI/CD pushing to AWS ECR, and a health check that gates on Triton kernel warmup.

**[edge-quantization-snapdragon](https://github.com/VrajPatel105/edge-quantization-snapdragon)**
Ran a decoder-only transformer entirely on the NPU of a Snapdragon 8 Elite Gen 5 with Qualcomm AI Hub, comparing FP16, W8A8, and W8A16. W8A8 cut latency from 2.5 ms to 1.4 ms, but top-1 agreement with FP32 fell to 24.6%. A PyTorch sensitivity sweep traced the loss to INT8 LayerNorm outputs and the residual stream, while INT8 linear layers alone were near-lossless.

**Earlier:** [NutriGrove](https://github.com/VrajPatel105/NutriGrove), a meal-recommendation app shipped on the App Store and Google Play (100+ users, 2nd place at a $12,000 AI + startup hackathon).

## Experience

**[General Dynamics Mission Systems](https://www.linkedin.com/in/vrajpatel105/)**, Capstone Team Member. Senior capstone at UMass Dartmouth sponsored by General Dynamics Mission Systems, applying deep learning to state estimation. Training and evaluating LSTM, GRU, and transformer models against classical baselines, including Kalman variants (KF, EKF, UKF) and particle filters (bootstrap, auxiliary, Rao-Blackwellized), within a custom simulation framework.

**[Fleet Robotics](https://www.linkedin.com/posts/vrajpatel105_finally-wrapping-up-my-summer-at-fleet-robotics-activity-7491448062242914306-sKL8)**, ML Engineer Intern (Summer 2026). Closed the sim-to-real gap for a hull-cleaning robot by finding that the MuJoCo simulator's noise averaged out instead of accumulating, rebuilding the noise model against AprilTag ground truth, and retraining the SAC navigation policy on the corrected simulator.

**[Blue Tech Externship](https://www.linkedin.com/posts/vrajpatel105_im-grateful-to-have-participated-in-the-activity-7441800946977931264-h3nf)**, Courage Builder Program (Mar 2026). Built a simulated autonomous UUV for sonar-based mine detection in MATLAB/Simulink; 1st place in the cohort.

**[SMAST, UMass Dartmouth](https://www.linkedin.com/posts/vrajpatel105_wrapping-up-an-incredible-micro-internship-activity-7394126929957773313-otMJ)**, Research Software Engineering Intern (Fall 2025). Refactored the R codebase for a NOAA-funded offshore wind and fishery survey project, parallelized spatial simulations for a 2x speedup, and packaged an internal R library.

**[SMAST, UMass Dartmouth](https://www.linkedin.com/posts/vrajpatel105_wrapping-up-an-incredible-summer-internship-activity-7359916390083936256-LVyu)**, Data Science Intern (Summer 2025). Automated analysis of 38 years of estuary data across 7 sites, cutting report turnaround from 3 weeks to 10 minutes.

## Open Source

**[vLLM](https://github.com/vllm-project/vllm)**: [PR #61901](https://github.com/vllm-project/vllm/pull/61901) (open, under review). Added config-time dtype validation for the Triton LoRA kernels, so an unsupported dtype (`float32` with `enable_lora`) is rejected at startup instead of crashing on an assert inside the `lora_expand` kernel.

**[CobraML2](https://github.com/govindansriram/CobraML2)**: [PR #39](https://github.com/govindansriram/CobraML2/pull/39) (open, under review). Added a GPT-2 benchmark and a configurable `use_naive_attention` option to compare attention implementations.

## Stack

Python, C++, CUDA, Triton, PyTorch, Nsight Compute/Systems, FastAPI, Docker, AWS, Snowflake

## Links

[Resume](https://vrajpatel105.github.io/resume.pdf) · [Portfolio](https://vrajpatel105.github.io/) · [LinkedIn](https://linkedin.com/in/vrajpatel105) · vrajpatel.jobs@gmail.com
