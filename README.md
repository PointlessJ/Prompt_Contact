# HOI Playground · G1 box carry

**Live demo: https://pointlessj.github.io/Prompt_Contact/**

A Unitree G1 walks to a box on the floor, picks it up, carries it to a target and puts it down. It only uses its head depth camera and proprioception: it is never told where the box is. MuJoCo and the policy run entirely in your browser (WebAssembly), at 50 Hz.

Things to try:
- Drag the box somewhere else while the robot walks to it.
- Double-click the floor to move the green target, even while the robot carries the box.
- Adjust **Walking pace (m/s)** (`0.30–0.60 m/s`, default `0.45 m/s`) on the default **Pace-controlled** policy to speed up or slow down approach and carrying in real time.
- Change the box: length, width and height separately, or the Cube / Random parcel / Flat presets, and its mass (0.5–5 kg). **Random** also draws a new cube (17–45 cm, 0.5–5 kg).
- Push the robot, soften the PD gains, or add actuation delay. These are the gaps between simulation and the real robot.
- Switch policies in the panel (`Pace-controlled`, `Soft-Prompt v2`, `v16 release`, `v3 release`) to compare pace control, contact-planner behavior, deployment robustness, heavy-box carrying, and post-release standing posture.

Keys: `R` reset · `N` random layout · `Space` pause · `.` single step · `P` push · `T` set target.

## Policies

Pick one in the panel, or add `?policy=pace_fix` / `?policy=softprompt_v2` / `?policy=v16` / `?policy=release` to the URL.

| Policy | What it is |
|---|---|
| `Pace-controlled (30k, EMA 0.998)` *(default, `pace_fix`)* | Stage 3 unified pace-controlled box-carry policy (`deploy_paceFix_30k_ema998`, branch `acp-s3-pace`, EMA 0.998), trained on parcels (25–45 × 18–32 × 13–30 cm) and cubes (17–61 cm) with live commanded walking pace `0.3–0.6 m/s` (default `0.45 m/s`, `hold_rise = 0.52 m`) |
| `Soft-Prompt v2 (36k, EMA 0.9995)` (`softprompt_v2`) | Differentiable Soft-Attention Contact-Prompt Planner (`deploy_v16ACPSoftPromptV2_36k_ema9995`, 6-input cloud cross-attention + object-point candidate contacts + manipulation prompt head, trained via SuperMimic world-model BPTT on top of frozen v16 policy, EMA 0.9995) |
| `v16 release (38k, EMA 0.9995)` (`v16`) | Stage 3 box-carry policy (`deploy_v16ACP001V1_38k_ema9995`, LayerNorm policy, EMA 0.9995), trained on cubes 0.5–1.8× (17–61 cm), 1–5 kg, with low hand friction (`mu_lo = 0.50`), 0–20 ms actuation delay, and soft PD randomization |
| `v3 release` (`release`) | Earlier Stage 3 box-carry policy, trained on cubes 0.5–1.8× (17–61 cm), 1–5 kg |

### Deployment bundle & ONNXRuntime closed-loop verification

Before exporting the browser graphs, the robot-side deployment bundles and standalone ONNX models were verified directly in closed-loop ONNXRuntime (`DEPLOY=drive`, zero drop vs PyTorch):

| ONNX closed-loop test (`DEPLOY=drive`) | `Pace-controlled (30k)` | `Soft-Prompt v2 (36k)` | `v16 release (38k)` |
|---|---:|---:|---:|
| `dep_base` (standard deploy) | **96.9%** *(64 ep, 24 s, `fell 0.0%`)* | **98.4%** (`nl 0.0%`, `fell 0.0%`) | **100.0%** (`nl 0.0%`) |
| `heavy` (2.0–3.2× box mass) | – | **100.0%** (`128/128`, `nl 0.0%`, `fell 0.0%`) | – *(PyTorch `5–6.9 kg`: 87.5%)* |
| `pd_soft` (0.6–0.75× PD gains) | – | **96.1%** (`nl 0.0%`, `fell 0.0%`) | **94.5%** (`nl 0.8%`) |
| `low_fric` (0.4–0.5 hand friction) | – | **99.2%** (`127/128`, `nl 0.0%`, `fell 0.0%`) | **98.4%** (`nl 0.8%`) |

- **`Pace-controlled (30k, EMA 0.998)` (`deploy_paceFix_30k_ema998`)**:
  - `deploy_bundle_paceFix_30k_ema998.pt` (39.59 MiB, SHA-256 `1037de8cc36d`)
  - `policy.onnx` (35.49 MiB, SHA-256 `0c5ed9cf60b2`), `planner.onnx` (4.08 MiB, SHA-256 `057d532df01a`)
  - Closed-loop `DEPLOY=parity` (16 ep, 14 s, noise-free): action max `7.6e-6`; command, image, object goal, cloud, task `0`.
  - ONNX `DEPLOY=drive` (64 ep, 24 s, noisy depth, nominal mount): **96.9%** placed (`0` falls; training-side reference 98.4%).
- **`Soft-Prompt v2 (36k, EMA 0.9995)` (`deploy_v16ACPSoftPromptV2_36k_ema9995`)**:
  - `deploy_bundle_softprompt_v2_36k_ema9995.pt` (40 MiB, SHA-256 `634eb73240e5`)
  - `policy.onnx` (36 MiB, SHA-256 `3073f95246d7`), `planner.onnx` (4.1 MiB, 6-input with `cloud`, SHA-256 `19b07e1b4879`)
  - `DEPLOY=parity` (`TF32=0`, 64 envs): action median `5.14e-6`, targets median `5.07e-7`, obs median `5.96e-7`; across 512 closed-loop ONNX episodes: **0% never-lifted (`nl = 0.0%`) and 0% fallen (`fell = 0.0%`)**.
- **`v16 release (38k, EMA 0.9995)` (`deploy_v16ACP001V1_38k_ema9995`)**:
  - `deploy_bundle_001_38k_ema9995.pt` (39.07 MiB, SHA-256 `29ede1a5cca8`)
  - `policy.onnx` (35.49 MiB, SHA-256 `eecd02c72397`), `planner.onnx` (3.57 MiB, SHA-256 `b1761c5e06b8`)
  - `DEPLOY=parity` (`TF32=0`, 64 envs): action median `4.29e-6`, targets median `4.77e-7`, obs max `1.91e-6`, task max `0.00e+00`.

## How it works

Every 20 ms control step runs three ONNX graphs, exported from the PyTorch training code:

| Graph | What it does |
|---|---|
| `models/sense.onnx` | The training simulator's sensing: a 32×18 head depth image and an 8×8 point cloud of the box (segmented, in the robot's heading frame) |
| `models/<policy>/init.onnx` | Task start: from the robot pose and the placement target to the task vector (the robot then estimates the box from its first view) |
| `models/<policy>/step.onnx` | One control step: observation history, contact planner, target latch, box estimate, task phase and whole-body policy. All controller state goes in and comes out |

Sensor noise follows the training model and is added in JavaScript: Gaussian noise, holes and edge dropout on the depth image, and ray-length noise, point dropout and frame dropout on the cloud. The depth and cloud delay is adjustable.

The policy is a contact-conditioned whole-body tracker. A contact planner chooses where hands, feet, knees, elbows and pelvis should touch next, among the points the camera sees. The tracker drives the body there. Training used a differentiable world model (SuperMimic-style imagination with backpropagation through time) in three stages: tracking reference motions, learning the planner from hindsight contacts, then box tasks without references.

Physics is the same MuJoCo scene used in training (`assets/scene/g1_box.xml`, position actuators with the training PD gains), stepped at 200 Hz with 4 substeps per control step. Before release, the exported browser graphs were verified against the PyTorch controller over 600–700 steps and run in Python MuJoCo and headless Chrome WebAssembly:
- `Pace-controlled`: 600-step parity (`pace = 0.45 m/s`) `med_act_diff = 4.77e-6` (`max 7.42e-5`), 700-step live-override parity (`pace = 0.35 m/s`) `med_act_diff = 4.29e-6` (`max 2.87e-5`, `end_dist = 0.008 m`); Python MuJoCo `16/16` at `0.45 m/s`, `8/8` at `0.30 m/s`, `8/8` at `0.60 m/s`, `13/16` on random 17–45 cm / 0.5–5 kg cubes (`0` falls across 48 episodes); headless Chrome **`6/6`** on random cubes (~9.4 ms/step).
- `Soft-Prompt v2`: 600-step parity `med_act_diff = 2.86e-6` (`max 1.11e-5`), `max_q_diff = 4.89e-6`; Python MuJoCo `16/16` 34 cm cubes, `16/16` random 17–45 cm / 0.5–5 kg cubes, `8/8` 0.7× soft PD, `8/8` 20 ms actuation delay (`0` falls across 56 episodes); headless Chrome **`6/6`** (~8.4 ms/step).
- `v16 release`: 600-step parity `max_act_diff = 8.64e-6`, `max_q_diff = 2.62e-6`; Python MuJoCo `16/16` 34 cm cubes, `16/16` random 17–45 cm / 0.5–5 kg cubes, `8/8` 20 ms delay, `7/8` 0.7× soft PD; headless Chrome **`6/6`** (~8.0 ms/step).

## Run locally

```sh
python -m http.server 8000
# open http://localhost:8000
```

Add `?test&episodes=5` to the URL to run episodes headless and print the results to the console (`&policy=pace_fix`, `&policy=softprompt_v2`, `&policy=v16`, or `&policy=release`, and `&box=cube` keeps the 34 cm cube instead of drawing random cubes).

## Credits

- [MuJoCo](https://github.com/google-deepmind/mujoco) (WebAssembly build `@mujoco/mujoco`), [onnxruntime-web](https://onnxruntime.ai/), [three.js](https://threejs.org/).
- Unitree G1 model and meshes: Unitree Robotics, via [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) and [mjlab](https://github.com/mujocolab/mjlab) (BSD-3-Clause / Apache-2.0).
- The idea of a browser playground comes from the [ACP playground](https://leggedrobotics.github.io/ACP/playground/) (ETH RSL, *Humanoid Mobility via Attention-based Contacts*). This page is an independent implementation.

---

**中文简介**：G1 只靠头部深度相机走到地上的箱子前，抱起，搬到绿色目标点放下。物理（MuJoCo）和策略（ONNX）全部在浏览器里运行。拖动箱子、双击地面换目标、分别改箱子的长宽高和重量（0.5–5 kg）、实时调节行走步速（0.30–0.60 m/s）、推机器人、调 PD 增益和执行延迟，都可以实时试。默认策略已升级为 **Pace-controlled（`paceFix 30k EMA 0.998`）**（支持 0.30–0.60 m/s 步速控制、包裹 + 方块混合训练，ONNX 闭环 `96.9%` / `0` 摔倒），同时保留 `Soft-Prompt v2 (36k)`、`v16 release (38k)` 与 `v3 release` 供下拉切换对比。
