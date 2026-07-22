import json, os, subprocess, sys, torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager


def _run_epoch_eval_inplace(eval_script: str, epoch_id: int, output_path: str,
                            accelerator: Accelerator):
    """In-place epoch eval: use the training model directly for inference.

    Machine 0 processes shard the benchmark and generate images in parallel
    using pipe() with torch.no_grad(). Machine 1 processes skip to the barrier.
    Then rank 0 runs scoring via subprocess. No model reloading needed.
    No shared filesystem required between machines.
    """
    benchmark_path = os.environ.get("EVAL_BENCHMARK",
                                    "/workspace/manga_sft/dataset/benchmark_0608.jsonl")
    dataset_base = os.environ.get("EVAL_DATASET_BASE",
                                  "/workspace/manga_sft/dataset/images")
    eval_root = os.environ.get("EVAL_OUTPUT_ROOT", "/workspace/manga_sft/eval_results")
    eval_output_dir = os.path.join(eval_root, f"epoch-{epoch_id}")
    summary_file = os.path.join(eval_root, "eval_summary.jsonl")
    width = int(os.environ.get("EVAL_WIDTH", "768"))
    height = int(os.environ.get("EVAL_HEIGHT", "1024"))
    num_inference_steps = int(os.environ.get("EVAL_STEPS", "40"))
    seed = int(os.environ.get("EVAL_SEED", "0"))

    panels_dir = os.path.join(eval_output_dir, "panels")

    # Phase 1: inference — only machine 0, machine 1 waits at barrier
    is_machine_0 = getattr(accelerator, "node_rank", 0) == 0

    with open(benchmark_path) as f:
        all_entries = [json.loads(line) for line in f if line.strip()]

    if is_machine_0:
        local_procs = accelerator.num_processes // int(os.environ.get("EVAL_NUM_MACHINES", "2"))
        local_rank = accelerator.process_index % local_procs
        my_entries = all_entries[local_rank::local_procs]
    else:
        my_entries = []

    if accelerator.is_main_process:
        os.makedirs(panels_dir, exist_ok=True)
        n_infer = len(my_entries) if is_machine_0 else 0
        print(f"[eval] epoch {epoch_id}: machine_0={is_machine_0}, "
              f"{n_infer} panels to generate", flush=True)
    accelerator.wait_for_everyone()

    # Access the pipeline from the training model (machine 0 only)
    if is_machine_0:
        unwrapped = accelerator.unwrap_model(accelerator.models[0])
        pipe = unwrapped.pipe
        pipe.eval()

        gen_ok, gen_fail = 0, 0
        for entry in my_entries:
            image_name = Path(entry["image"]).stem
            save_path = os.path.join(panels_dir, f"{image_name}.png")
            ref_paths = [os.path.join(dataset_base, p) for p in entry.get("edit_image", [])]
            ref_images = []
            for p in ref_paths:
                if os.path.exists(p):
                    ref_images.append(Image.open(p).convert("RGB"))
            try:
                with torch.no_grad():
                    kwargs = dict(seed=seed, num_inference_steps=num_inference_steps,
                                  height=height, width=width, zero_cond_t=True,
                                  cfg_scale=1.0,
                                  progress_bar_cmd=lambda x: x)
                    if ref_images:
                        img = pipe(entry["prompt"], edit_image=ref_images, **kwargs)
                    else:
                        img = pipe(entry["prompt"], **kwargs)
                img.save(save_path)
                gen_ok += 1
            except Exception as e:
                gen_fail += 1
                print(f"[eval] {image_name} ERROR: {e}", flush=True)

        pipe.train()
        if accelerator.is_main_process:
            print(f"[eval] inference done: {gen_ok} ok, {gen_fail} fail", flush=True)

    # Wait for machine 0 to finish inference
    accelerator.wait_for_everyone()

    # Phase 2: scoring — rank 0 spawns subprocess (Gemini API, CPU-only)
    if accelerator.is_main_process:
        env_score = os.environ.copy()
        cmd_score = [
            sys.executable, eval_script,
            "--checkpoint", "unused",
            "--epoch", str(epoch_id),
            "--output-dir", eval_output_dir,
            "--mode", "score",
            "--summary-file", summary_file,
        ]
        score_log = os.path.join(eval_output_dir, "score.log")
        try:
            with open(score_log, "w") as log_file:
                subprocess.run(cmd_score, env=env_score, stdout=log_file, stderr=subprocess.STDOUT)
            print(f"[eval] scoring done for epoch {epoch_id}", flush=True)
        except Exception as e:
            print(f"[eval] scoring FAILED for epoch {epoch_id}: {e}", flush=True)

    # Wait for scoring before resuming training
    accelerator.wait_for_everyone()


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            eval_script = os.environ.get("EVAL_SCRIPT")
            if eval_script:
                _run_epoch_eval_inplace(eval_script, epoch_id, model_logger.output_path, accelerator)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
