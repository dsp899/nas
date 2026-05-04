LOT = {
    "search_space": {
        "layers": [1, 2, 3],
        "rnn": ["lstm", "gru"],
        "units_0": [64, 128, 256, 512, 1024],
        "units_1": [0, 64, 128, 256, 512, 1024],
        "units_2": [0, 64, 128, 256, 512, 1024],
        "direction": ["unidirectional", "bidirectional"],
        "memory_mode": ["none", "carry_forward"],
        "seq": [3, 6, 9, 12],
        "head_units": [128, 256, 512, 1024],
        "video_decision": ["average", "majority", "max_prob"],
        "video_decision_input": ["clip_logits", "clip_embeddings"],
    },
    "generation_seed": 1234,
    "requested_count": 200,
    "experiment": {
        "dataset_profile": "ucf101",
        "num_classes": 101,
        "feature_dim": 512,
        "video_steps": 36,
    },
}

STORAGE = {
    "persist_float_model": False,
    "persist_tflite_model": True,
}

RESOURCE_MANAGER = {
    "max_ram_fraction": 0.70,
    "ram_reserve_mb": 32768,
    "generation_estimated_worker_ram_mb": 2200,
    "benchmark_estimated_worker_ram_mb": 5500,
}

GENERATION = {
    "runtime": {
        "jobs": "auto",
        "worker_max_tasks": 4,
        "intra_op_threads": 1,
        "inter_op_threads": 1,
    }
}

BENCHMARK = {
    "signature": {
        "feature_source": "synthetic",
        "num_videos": 8,
        "feature_seed": 1234,
        "distribution": "normal",
        "warmup_runs": 5,
        "steady_runs": 10,
        "threads": 1,
        "hardware_target": "fenix_cpu",
        "experiment_name": "bench_default",
    },
    "runtime": {
        "runtime": "tflite",
        "jobs": "auto",
        "cpu_policy": "none",
        "cpu_reserve_cores": 1,
        "worker_max_tasks": 4,
    }
}

DATASET = {}
SPLIT = {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15, "seed": 1234}
GNN = {}
