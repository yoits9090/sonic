import platform, os, sys, json
info = {"platform": platform.platform(), "python": sys.version.split()[0]}
try:
    with open("/proc/cpuinfo") as f:
        cpu = f.read()
    models = set(l.split(":")[1].strip() for l in cpu.splitlines() if l.startswith("model name"))
    info["cpu"] = list(models)
    info["cores"] = cpu.count("processor")
except Exception as e:
    info["cpu_err"] = str(e)
info["mem_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
import numpy as np
info["numpy"] = np.__version__
try:
    info["blas"] = np.__config__.blas_opt_info.get("libraries", None) if hasattr(np.__config__, "blas_opt_info") else None
except Exception:
    pass
# quick matmul speed probe: 512x512 fp32
import time
a = np.random.rand(512, 512).astype(np.float32); b = np.random.rand(512, 512).astype(np.float32)
a @ b  # warmup
t0 = time.perf_counter()
for _ in range(50): a @ b
info["matmul512_gflops"] = round(2 * 512**3 * 50 / (time.perf_counter() - t0) / 1e9, 1)
print(json.dumps(info))
