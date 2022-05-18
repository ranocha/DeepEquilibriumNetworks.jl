import subprocess

TYPES = ("SKIPV2", "SKIP", "VANILLA")
EXPT_IDS = {
    "SKIPV2": [18010267, 18010268, 18010269],
    "SKIP": [18014477, 18010271, 18010272],
    "VANILLA": [18014141, 18014142, 18014144],
}

for t in TYPES:
    for id in EXPT_IDS[t]:
        subprocess.run(["rsync", "-vPrR", f"supercloud:research/FastDEQ.jl/logs/data-CIFAR10_type-{t}_size-TINY_discrete-false_jfb-false/./{id}/", "cifar10/tiny"])