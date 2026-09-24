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

**[Massachusetts Housing Price Predictor](https://github.com/VrajPatel105/Massachusetts-Housing-Recommender-System)**
End-to-end ML pipeline: a Zillow scraper collecting Massachusetts listings with 40+ features, cleaning and feature engineering, an ExtraTrees regressor reaching R² ≈ 0.854 on price prediction, and a property recommender deployed in Streamlit. The cleaned dataset is published on Kaggle.

## Experience

**[General Dynamics Mission Systems](https://www.linkedin.com/in/vrajpatel105/)**, Capstone Team Member. Senior capstone at UMass Dartmouth sponsored by General Dynamics Mission Systems, applying deep learning to state estimation. Training and evaluating LSTM, GRU, and transformer models against classical baselines, including Kalman variants (KF, EKF, UKF) and particle filters (bootstrap, auxiliary, Rao-Blackwellized), within a custom simulation framework.

**[Fleet Robotics](https://www.linkedin.com/posts/vrajpatel105_finally-wrapping-up-my-summer-at-fleet-robotics-activity-7491448062242914306-sKL8)**, ML Engineer Intern (Summer 2026). Closed the sim-to-real gap for a hull-cleaning robot by finding that the MuJoCo simulator's noise averaged out instead of accumulating, rebuilding the noise model against AprilTag ground truth, and retraining the SAC navigation policy on the corrected simulator.

**[Blue Tech Externship](https://www.linkedin.com/posts/vrajpatel105_im-grateful-to-have-participated-in-the-activity-7441800946977931264-h3nf)**, Courage Builder Program (Mar 2026). Built a simulated autonomous UUV for sonar-based mine detection in MATLAB/Simulink; 1st place in the cohort.

**[SMAST, UMass Dartmouth](https://www.linkedin.com/posts/vrajpatel105_wrapping-up-an-incredible-micro-internship-activity-7394126929957773313-otMJ)**, Research Software Engineering Intern (Fall 2025). Refactored the R codebase for a NOAA-funded offshore wind and fishery survey project, parallelized spatial simulations for a 2x speedup, and packaged an internal R library.

**[SMAST, UMass Dartmouth](https://www.linkedin.com/posts/vrajpatel105_wrapping-up-an-incredible-summer-internship-activity-7359916390083936256-LVyu)**, Data Science Intern (Summer 2025). Automated analysis of 38 years of estuary data across 7 sites, cutting report turnaround from 3 weeks to 10 minutes.

## Stack

Python, C++, CUDA, Triton, PyTorch, Nsight Compute/Systems, FastAPI, Docker, AWS, Snowflake

## Links

[Resume](https://vrajpatel105.github.io/resume.pdf) · [Portfolio](https://vrajpatel105.github.io/) · [LinkedIn](https://linkedin.com/in/vrajpatel105) · vrajpatel.jobs@gmail.com
