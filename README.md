cmake .. -DQWEN35MOE_CUDA=ON


cmake -S . -B build && make -C build -j4
./build/test_qwen35moe --model /path/to/model.gguf --prompt "Hello, explain quantum computing" --n-predict 256 --temp 0.7


./test_qwen35moe --model /home/xc/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf --prompt "Hello, explain quantum computing" --n-predict 256 --temp 0.7