# effdl26-liger-kernel-report

Materials for small report about [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/tree/main) framework made by me during Efficient DL Systems course (spring 26') in Yandex School of Data Analysis.

- To set up environment needed for work, use [uv](https://github.com/astral-sh/uv) package manager and command `uv sync`.
- To run benchmarks use `./run_train_benchmark.sh` after setting the environment up.
- All other code blocks are located in `report.ipynb` (which again should be run under the environment specified by `uv.lock`)
- `train_llama.py` contains flexible (as flexibled as I succeeded to write) code for training Llama-3.1-8B for several iterations to profile and measure the performance of model under the changes from Liger-Kernel.

Some introductionary slides from my presentation can be found [there](https://docs.google.com/presentation/d/1gl-TKhdUpFACFOxT6EF9ppM95-ro3GAtVvLrKVTCeJ0/edit?usp=sharing).

Good luck!
