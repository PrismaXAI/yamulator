### Introduction

`yamulator` is a simulator for bimanual YAM rigs designed to be used as a supplementary eval loss during training as well as a tool for autonomous agent-driven ablation studies during rollout.
The rendering code leverages physically-based models for materials, lighting, and contact, with a GPU implementation of the WDAS Principled BRDF for appearance control, a fast OptiX path tracer which automatically leverages RTX on supported GPUs, and a simple, extensible Mujoco-derived contact model for gripper interactions. Because OptiX does not require a raster engine, the simulator will run with high performance on raster-less training nodes, allowing large scale (10's or 100's of thousands) of trials during ablation to discover trends in model performance.

### Usage and Limitations

Like all of our open-source projects, `yamulator` is designed to be deployed by an agent. Simply point your frontier agent (tested on Claude Fable 5.1, likely works on Astra) at this repository and ask it to clone and deploy; the repo is extensively documented in a way which is easy for agents to understand. The release comes with a simple pick and place example -- your agent should
be able to easily construct new geometries. We've found that this is better than trying to write a general scene loader -- rigid bodies are trivial anyway and accurate deformable physics are unlikely to run out of the box.

It is important to understand the limitations of the simulator to properly use it:

- The shading model is state of the art but not based on measured material properties. As you probably are not interested in measuring the BRDF of your targets, we follow industry best-practice and use an easily-controllable parametric approximation. The WDAS BRDF is very good and is used in commercial product and architectural rendering packages as well as nearly every Hollywood VFX tool.
- The light transport model is an unbiased unidirectional path tracer and should be 100% accurate. The renderer uses neural denoising to reduce the required sample count. OptiX neural denoising is very good but if you feel like there are rendering pathologies you can always up the base sample count to do A/B's.
- The physics model is OK. By default, collisions for the arms themselves are turned off -- only scene objects and the end effectors participate. The end effector uses a good friction-cone based model with optional compliant pads.
- Because the physics model is only "OK", you should not use this simulator to collect data. In particular, we expect accuracy with complex concave geometries to be subpar and with deformable objects to be nonexistent.
- In general, we find accuracy in sim to be monotonic with respect to accuracy in real and for large gaps in sim to also exist in real. Additionally, the sim usually outperforms real. This makes the tool useful for hunting for hyperparameters beyond just eval loss or chasing down bugs or training pathologies (e.g. camera errors, out of distribution backgrounds, ...)


### About Us

`yamulator` is brought to you by [PrismaX](http://prismax.ai). We provide end-to-end services for data collection and model rollout on YAM and other popular embodiments, with over 6000 hours a mnth of capacity across the most challenging dexterous tasks in the industry, operators with thousands of hours of experience, and sensor, end effector, and scenario customization to meet you and your researchers' needs.
