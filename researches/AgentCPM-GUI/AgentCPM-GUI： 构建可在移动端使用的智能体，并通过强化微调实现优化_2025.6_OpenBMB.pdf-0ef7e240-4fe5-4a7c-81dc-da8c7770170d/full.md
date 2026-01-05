# AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning

Zhong Zhang $^{1*}$ , Yaxi Lu $^{1*}$ , Yikun Fu $^{1\dagger}$ , Yupeng Huo $^{2}$ , Shenzhi Yang $^{2}$ , Yesai Wu $^{1}$ , Han Si $^{1\dagger}$   
Xin Cong $^{1}$ , Haotian Chen $^{1}$ , Yankai Lin $^{2\dagger}$ , Jie Xie $^{1}$ , Wei Zhou $^{1}$ , Wang Xu $^{1}$ , Yuanheng Zhang $^{1\dagger}$   
Zhou Su $^{3}$ , Zhongwu Zhai $^{3}$ , Xiaoming Liu $^{3}$ , Yudong Mei $^{3}$ , Jianming Xu $^{3}$ , Hongyan Tian $^{3}$   
Chongyi Wang $^{3}$ , Chi Chen $^{1}$ , Yuan Yao $^{1}$ , Zhiyuan Liu $^{1\dagger}$ , Maosong Sun $^{1\dagger}$

$^{1}$ Tsinghua University  $^{2}$ Renmin University of China  $^{3}$ ModelBest Inc. zhongzhang@tsinghua.edu.cn lyx23@mails.tsinghua.edu.cn

![](images/a7d220e7ec8b3dfe1be58904a1d52db9c931f0fb3b08c5197a3a4fec927d91d1.jpg)

https://github.com/OpenBMB/AgentCPM-GUI

# Abstract

The recent progress of large language model agents has opened new possibilities for automating tasks through graphical user interfaces (GUIs), especially in mobile environments where intelligent interaction can greatly enhance usability. However, practical deployment of such agents remains constrained by several key challenges. Existing training data is often noisy and lack semantic diversity, which hinders the learning of precise grounding and planning. Models trained purely by imitation tend to overfit to seen interface patterns and fail to generalize in unfamiliar scenarios. Moreover, most prior work focuses on English interfaces while overlooks the growing diversity of non-English applications such as those in the Chinese mobile ecosystem. In this work, we present AgentCPM-GUI, an 8B-parameter GUI agent built for robust and efficient on-device GUI interaction. Our training pipeline includes grounding-aware pre-training to enhance perception, supervised fine-tuning on high-quality Chinese and English trajectories to imitate human-like actions, and reinforcement fine-tuning with GRPO to improve reasoning capability. We also introduce a compact action space that reduces output length and supports low-latency execution on mobile devices. AgentCPM-GUI achieves state-of-the-art performance on five public benchmarks and a new Chinese GUI benchmark called CAGUI, reaching  $96.9\%$  Type-Match and  $91.3\%$  Exact-Match. To facilitate reproducibility and further research, we publicly release all code, model checkpoint, and evaluation data.

# 1 Introduction

The rapid advancements in Large Language Models (LLMs) and Multimodal Large Models (MLLMs) have catalyzed a new era of autonomous AI agents (Zhao et al., 2023; Wang et al., 2024b). These agents are increasingly capable of understanding complex instructions (Ouyang et al., 2022; Qian et al., 2024), performing multi-step planning (Huang et al., 2024), and interacting with external tools or environments (Qin et al., 2024, 2025a). A critical frontier for deploying these intelligent agents in practical, human-centric applications is enabling them to proficiently operate Graphical User Interfaces (GUIs) (Wang et al., 2024c; Nguyen et al., 2024; Zhang et al., 2024a), particularly within the ubiquitous Android ecosystem, where they serve as the primary interaction layer for a vast array of daily digital tasks. Empowering LLM agents to seamlessly navigate and manipulate these mobile GUIs is essential for transforming them into truly versatile digital assistants capable of automating a wide spectrum of tasks on smartphones, thereby enhancing user productivity and accessibility.

Early GUI agents emerged when Vision-Language Models (VLMs) had limited ability in reliably control GUI widgets. To compensate, researchers augmented model inputs with structured metadata, such as Android view hierarchies and system APIs, and even off-loaded perception and planning to more capable external VLMs (e.g., GPT-4o (Hurst et al., 2024)), thereby improving widget grounding and action execution (Zhang et al., 2025; Chen et al., 2025a; Chen & Li, 2024; Zheng et al., 2024; Kim et al., 2023; Wang et al., 2024a). Although effective, these hybrid pipelines propagated errors from cross-modal mismatches, incurred round-trip latency, and depended on metadata that many apps do not expose, creating significant challenges for generality and scalability. Recent GUI agents have advanced to resolving interface elements directly from raw pixels, enabling a single end-to-end model to match or even surpass earlier hybrid approaches (Hong et al., 2024; Cheng et al., 2024; Qin et al., 2025b; Xu et al., 2024; Wu et al., 2025; Lin et al., 2024; Zhang & Zhang, 2024). This shift positions purely visual, end-to-end modeling as the most scalable paradigm.

Despite significant progress, current visual GUI agents still face several challenges: (1) Data quality and scale. High-quality, fine-grained interaction trajectories that capture realistic user behavior in diverse mobile apps are notoriously difficult to collect at scale. Most publicly available datasets either rely on synthetic generation or emulator-based recordings, both of which can introduce noise and lack semantic diversity. Such imperfect supervision limits the agent's ability to learn precise widget grounding, compositional reasoning, and long-horizon action planning. (2) Reasoning generalization. GUI agents that are trained solely via imitation learning tend to overfit to interface patterns, resulting in brittle planning and poor generalization when task instructions deviate from seen templates or when UI layouts exhibit minor variations. (3) Language and regional coverage. Current research concentrates almost exclusively on English GUIs, paying limited attention to the rapidly growing and diverse Chinese mobile ecosystem, whose interface design conventions and linguistic cues differ substantially. These differences limit the generalizability of current agents in multilingual and culturally diverse settings.

To address these challenges, we propose AgentCPM-GUI, a VLM-based agent for mobile GUI understanding and interaction. The key features of this work are as follows.

- High-quality training data. We curate a large-scale corpus of 55K trajectories with 470K steps, encompassing a wide variety of Chinese Android apps via targeted collection and meticulous annotation. To enhance generalization and mitigate overfitting, we further incorporate and rigorously de-duplicate multiple public English Android datasets. The resulting unified dataset supports effective training, enabling robust cross-lingual and cross-app behavior modeling.  
- Progressive training for perception, imitation, and reasoning. We adopt a three-stage progressive training pipeline to equip the agent with strong GUI understanding and reasoning capabilities, consisting of grounding-aware pre-training to enhance visual perception; supervised fine-tuning (SFT) to establish a reliable behavioral prior; and reinforcement fine-tuning (RFT) (OpenAI, 2024; Shao et al., 2024; Trung et al., 2024) to further strengthen reasoning ability, enabling robust performance on long-horizon and compositional tasks. In addition, we optimize the training framework with asynchronous rollout and load balancing to support scalable reinforcement learning.  
- Edge device oriented design. To reduce decoding overhead, we carefully select action tokens to avoid unnecessary token fragmentation and adopt a compact JSON-based action format, resulting in an average output length of just 9.7 tokens per action. While prior works largely overlook redundancy in action space design, our concise representation significantly improves runtime efficiency, enabling smooth and responsive on-device execution.  
- Comprehensive benchmarking. We evaluate AgentCPM-GUI on the widely used English GUI agent benchmarks: AndroidControl (Li et al., 2024a), GUI-Odyssey (Lu et al., 2024a), and AITZ (Zhang et al., 2024b). In addition, we introduce CAGUI, the first large-scale Chinese Android GUI benchmark. CAGUI is a representative subset of our corpus designed for public evaluation. AgentCPM-GUI achieves new state-of-the-art performance across all datasets, demonstrating robust multilingual and cross-app generalization.

To support community research and ensure reproducibility, we open-source all model, training and evaluation code, along with the CAGUI benchmark. We believe this work establishes a solid foundation for advancing multilingual GUI agents in real-world applications.

# 2 Related Work

# 2.1 Datasets and Benchmarks

To facilitate progress in the field of GUI Agents, a number of datasets and benchmarks have been developed. Some of these datasets are specifically constructed for grounding tasks, which aim to associate natural language instructions with corresponding GUI widgets on the screen. Representative examples include Mind2Web (Deng et al., 2023), ScreenSpot/-Pro (Cheng et al., 2024; Li et al., 2025), OS-ATLAS (Wu et al., 2025), GUICourse (Chen et al., 2025b) and UGround (Gou et al., 2025). Others are constructed for GUI agent tasks, where tasks are represented as sequences of action-observation pairs, such as AITW (Rawles et al., 2023), AITZ (Zhang et al., 2024b), AndroidControl (Li et al., 2024a), GUI-Odyssey (Lu et al., 2024a), AMEX (Chai et al., 2024) and AndroidWorld (Rawles et al., 2024). Despite their contributions, existing datasets predominantly focus on English GUIs, limiting the development of GUI agents in non-English environments. To bridge this gap, we introduce CAGUI, a new benchmark covering both grounding and agent tasks in realistic Chinese mobile apps, enabling more robust assessment in multilingual settings.

# 2.2 VLM-based GUI Agents

GUI agents have moved from API-based agent frameworks to vision-only, end-to-end agent models. Early work such as AppAgent (Zhang et al., 2025; Li et al., 2024b) fused GPT-4(V) (OpenAI, 2023) with Android XML trees to build manuals for each screen. Mobile-Agent (Wang et al., 2024a) and SeeAct (Zheng et al., 2024) couple grounding modules with a reasoning LLM to decompose goals, ground elements and retry when actions fail. Newer agents place all perception inside a VLM that operates on raw screenshots (Zhang & Zhang, 2024; Lu et al., 2024a; Lin et al., 2024; Yang et al., 2024). Recent efforts focus on scaling both models and data. CogAgent (Hong et al., 2024), Aguvis (Xu et al., 2024) and UI-TARS (Qin et al., 2025b) pre-train high-capacity (7B-70B) VLMs on millions of screenshots for GUI grounding, then fine-tune models on interaction trajectories to enhance planning and reasoning capabilities. Notably, OS-Atlas (Wu et al., 2025) scales pre-training to 2.3M cross-platform screens with 13M labeled elements and releases the data and model. OS-Genesis (Sun et al., 2024) automates data collection by letting a seed agent explore apps, record its own steps, and filter them into high-quality trajectories used for later training.

# 2.3 Reinforcement Learning for GUI Agents

Most GUI agents rely on behavior cloning from fixed demonstrations, making their reasoning capability degenerate into pattern matching. Introducing reinforcement feedback lets agents learn from interaction and rewards, improving reasoning generalization.

DigiRL (Bai et al., 2024) leverages a two-stage pipeline that first applies offline RL for initial policy learning, followed by online RL to allow the agent to improve through real-time exploration and feedback. DistRL (Wang et al., 2025) proposes a scalable asynchronous reinforcement learning framework for on-device agents, combining centralized policy training with decentralized data collection and a custom off-policy algorithm to improve training efficiency and task success in dynamic mobile environments. Digi-Q (Bai et al., 2025) trains a Q-value function offline on frozen VLM representations and employs a best-of-N action sampling strategy to optimize policies without requiring additional environment interactions.

Recently, RFT-tuned VLMs have shown promising performance on various vision tasks (Zhai et al., 2024; Liu et al., 2025b; Tan et al., 2025; Huang et al., 2025; Zhou et al., 2025), and extending this strategy to GUI agents has yielded notable improvements. UI-R1 and GUI-R1 (Lu et al., 2025; Xia & Luo, 2025) adopt simple rule-based reward functions to assess the correctness of actions, using these signals to fine-tune the agent's capabilities through reinforcement. InfGUI-R1 (Liu et al., 2025a) proposes a reasoning-centric progressive training paradigm, transforming GUI agents from reactive executors into deliberative reasoners. AppVLM (Papoudakis et al., 2025) adopts an RFT framework that iteratively collects successful trajectories through online interaction and refines the policy via supervised fine-tuning, enabling efficient policy improvement without relying on reinforcement learning algorithms.

![](images/497d8f06ebb05e55fa3267cf8b5e85e39febf38e10ee056e470e73dcf4ba84cb.jpg)  
Figure 1: Overview of our training framework.

# 3 Method

# 3.1 Architecture Overview

To train a VLM capable of performing GUI-based interactions, we adopt a three-stage training framework that incrementally builds the model's capabilities from perception to action. Each stage targets a distinct subskill essential for robust and generalizable GUI operation. Our GUI agent is built upon MiniCPM-V (Yao et al., 2024), a lightweight vision-language model with 8B parameters that supports efficient visual grounding and instruction following, making it particularly well-suited for mobile-centric applications.

Stage I: Visual Perception and Grounding. In the first stage, we focus on enhancing the model's perceptual and grounding abilities. We curate a dataset of vision-language alignment tasks, including OCR and widget-localization, to help the model learn fine-grained spatial and semantic correspondences between GUI widgets and their descriptions. This establishes a strong foundation for the subsequent training stages.

Stage II: Supervised Imitation Learning. In the second stage, we collect GUI task execution trajectories paired with natural language instructions. Based on the model trained in Stage I, we conduct supervised fine-tuning to teach the model to generate valid and context-aware actions. This stage enables the model to imitate human-like action sequences when given a query, bridging the gap between perception and action.

Stage III: Reinforcement Fine-tuning. In the final stage, we apply RFT to further improve the model's reasoning and decision-making capabilities in complex GUI environments. Using the collected trajectories as initial demonstrations, we train the model with Group Relative Policy Optimization (GRPO) (Shao et al., 2024). This process optimizes the model's reasoning and action capabilities by rewarding correct and goal-directed action sequences, thereby pushing the model beyond simple imitation into more robust autonomous planning and adaptive behavior.

This staged curriculum progressively transforms a general-purpose vision-language model into a GUI-capable agent, leveraging a structured combination of perception-level training, supervised action learning, and reinforcement-based policy optimization.

# 3.2 Action Space Design

A well-designed, unified, and language model-friendly action space is crucial for enabling models to understand and generalize behaviors effectively, as highlighted in numerous prior works. Inspired by these studies, we propose an action space that both reduces generation length and supports compositional actions. Our action space consists of six primary atomic action types listed as follows, and example actions are listed in Table 1.

- POINT: Allows the model to locate a coordinate for performing operations. It takes a tuple  $(x, y)$  of integers in the range  $[0, 1000]$ , normalized with  $[0, 0]$  at the top-left corner of the current window and  $[1000, 1000]$  at the bottom-right. By default, this action performs a tap at the specified coordinate. It can also be combined with to or duration to express a swipe gesture or a long-press action, respectively.  
- to: Enables scrolling in the window. This action can either specify a direction from {"up", "down", "left", "right"} or define a swipe gesture when combined with a POINT target coordinate.  
- TYPE: Allows the model to input text into the current window. This action takes a string as its argument.  
- PRESS: Triggers special device keys, including "HOME", "BACK", "ENTER". These keys represent common actions across devices and provide a concise way to express frequent operations.  
- STATUS: Enables the model to update the current task status, including "continue", "finish", "satisfied", "impossible", "interrupt", "need_feedback". This action is used to communicate the state of execution. The default status is "continue" and can be omitted.  
- duration: Specifies the length of time (in milliseconds) that an action should persist. This parameter can be used independently to express an idle wait or combined with other actions (e.g., POINT) to indicate a long press or swipe duration.

To enhance efficiency and reduce token overhead during generation, we adopt a compact JSON representation that eliminates unnecessary whitespace between control characters. This contributes to a low average token cost of 9.7 per action, helping to reduce latency and improve responsiveness on edge devices.

Table 1: Example actions of AgentCPM-GUI.  

<table><tr><td>Example Actions</td><td>Purpose</td></tr><tr><td>{&quot;POINT&quot;:[480,320]}</td><td>Single tap at the normalized screen coordinate.</td></tr><tr><td>{&quot;POINT&quot;:[480,320],&quot;duration&quot;:1000}</td><td>Touch-and-hold (long press) at the coordinate.</td></tr><tr><td>{&quot;POINT&quot;:[500,200],&quot;to&quot;:&quot;down&quot;}</td><td>Swipe to a direction or another coordinate.</td></tr><tr><td>{&quot;PRESS&quot;:&quot;HOME&quot;}</td><td>Trigger a hardware or navigation key.</td></tr><tr><td>{&quot;TYPE&quot;:&quot;Hello,world!&quot;}</td><td>Insert the given text at the current input focus.</td></tr><tr><td>{&quot;duration&quot;:500}</td><td>Idle for the specified time in milliseconds.</td></tr><tr><td>{&quot;STATUS&quot;:&quot;finish&quot;}</td><td>Task status, can stand alone or with an action.</td></tr></table>

# 3.3 Stage I: Visual Perception and Grounding

For grounding pre-training, we collect Android GUI data by sampling examples from several open-source corpora (AITZ (Zhang et al., 2024b), GUICourse (Chen et al., 2025b), OS-Atlas (Wu et al., 2025), UGround (Gou et al., 2025), ScreenSpot (Cheng et al., 2024)) and additional screenshots from our collected Chinese app data. Each image is formulated as either an OCR task that asks the model to write the text in a marked region, or a widget-localization task that asks it to output the bounding box coordinate of a referenced UI element.

Grounding batches mix in  $50\%$  general multimodal SFT data (e.g., Chat, VQA, Multimodal Reasoning) (Yao et al., 2024), which regularizes the vision module while letting it absorb GUI-specific cues. In total, the grounding pre-training dataset comprises 12M samples.

This pre-training stage plays a crucial role in establishing the model's low-level perceptual and grounding abilities. We observe that, after this stage, the model demonstrates strong proficiency in identifying and locating GUI widgets, especially in accurately predicting coordinates based on visual cues. However, the model at this point still struggles to generate well-formed function calls or to reason over action types, indicating limited understanding of higher-level task semantics and planning. These capabilities are further enhanced in the subsequent SFT and RFT stages.

# 3.4 Stage II: Supervised Imitation Learning

Due to the scarcity of high-quality open-source datasets for Chinese Android apps, we constructed a large-scale, high-fidelity dataset of GUI interaction trajectories to support supervised imitation learning. The corpus covers over 30 mainstream Chinese apps, spanning eight functional domains: life services, e-commerce, navigation, social, video, music/audio, reading/learning, and productivity. This ensures that the agent is exposed to a wide spectrum of UI layouts, widget types, and task intents. In total, we obtained 55K complete task trajectories comprising 470K atomic steps, approximately 8.5 steps per trajectory. The data curation process is as follows:

- **Query Generation.** We first designed parameterized instruction templates for each app to reflect its core user intents. Slot values such as store names and quantities were then filled using GPT-4o. Human annotators reviewed the resulting queries, removing errors and duplicates. Finally, another GPT-4o pass paraphrased the verified queries into diverse surface forms, broadening lexical coverage and mitigating over-fitting.  
- Trajectory Collection. We collected trajectories on physical Android phones using a custom data logger. Running on real devices bypassed the feature limitations of emulators and preserved sensor signals like GPS and accelerometer data. Annotators received a queued query, completed the task, and confirmed each deliberate action. The logger saved only these validated taps, long-presses, swipes, text inputs, and their associated UI metadata while removing the spurious events that screen-capture pipelines often introduce.  
- Quality Assurance. Publicly available Android GUI corpora like AITW (Rawles et al., 2023) and Android-Control (Li et al., 2024a) contain non-negligible mislabeled actions and other annotation errors introduced by unrestricted screen-recording pipelines. To ensure data quality, we implemented two measures. First, our data logger records actions only if they are explicitly confirmed by annotators, preventing accidental gestures from being logged. Second, we apply post-hoc filtering to remove trajectories that miss essential steps, fail to complete the intended task, or duplicate existing examples.

To warm up the model for reasoning, we introduced preliminary thought generation at the SFT stage. We annotated 24K interaction trajectories from Chinese apps using GPT-4o to supply intermediate reasoning supervision. The dataset was further augmented with English thought-annotated data from AITZ (Zhang et al., 2024b) and AndroidControl (Li et al., 2024a). This warm-up is essential because without it, the model failed to generate reasoning traces during the RFT stage. In addition, to enable controlled thought generation at inference time and the RFT stage, samples with thought traces adopted a schema where the thought field was marked as "required", whereas those without were marked as "optional".

In order to enhance cross-lingual generalization and reduce over-fitting, we augmented our Chinese corpus with publicly available English-language datasets: AITW (Li et al., 2024a), AITZ (Zhang et al., 2024b), AMEX (Chai et al., 2024), AndroidControl (Li et al., 2024a), and GUI-Odyssey (Lu et al., 2024a). Since AITW is internally redundant, we performed intra-query de-duplication. For each trajectory, we extracted ResNet-50 features from its screenshots and averaged them to produce a trajectory embedding. We then grouped trajectories by shared query and, within each group, removed those whose cosine similarity to any previously retained sample exceeded a fixed threshold. This retained approximately  $40\%$  of the original data.

Empirically, training solely on GUI-interaction data led to a pronounced mode collapse during the subsequent RFT stage, manifesting as impoverished and repetitive reasoning thoughts. To mitigate this, we mixed  $50\%$  general multimodal SFT data into training batches, which helped stabilize policy optimization. The SFT data comprises a mix of single-turn (system-user-assistant) and multi-turn dialogues. For multi-turn examples, we retained only the last three turns of user-assistant interaction to provide sufficient conversational context while keeping input sequences within tractable length limits. In total, 6.9M instances were used for the SFT stage.

# 3.5 Stage III: Reinforcement Fine-tuning

We introduce an RFT stage to improve the agent's reasoning ability. To make RFT practical at scale, we further develop a training framework which supports asynchronous rollout and two levels of load balancing to improve efficiency and scalability across distributed environments.

# 3.5.1 Algorithmic Design

GRPO Algorithm. We conduct RFT based on the GRPO (Shao et al., 2024) algorithm. GRPO replaces the value critic of PPO (Schulman et al., 2017) with a group-wise comparison of candidate completions. Given a query  $q$ , the current policy  $\pi_{\theta_{\mathrm{old}}}$  samples  $N$  responses  $\{o_1, \dots, o_N\}$ . Each response is assigned a scalar task reward  $\{r_1, \dots, r_N\}$ . Rewards are normalised within the group to produce variance-reduced advantages:

$$
\hat {A} _ {i} = \frac {r _ {i} - \operatorname {m e a n} (\mathbf {r})}{\operatorname {s t d} (\mathbf {r})}, \quad \mathbf {r} = \left\{r _ {1}, \dots , r _ {N} \right\}. \tag {1}
$$

The policy is then updated using a clipped objective with KL divergence penalty:

$$
\mathcal {J} _ {\mathrm {G R P O}} (\theta) = \mathbb {E} _ {q \sim P (Q), \{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {o l d}} (O | q)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| o _ {i} |} \sum_ {t = 1} ^ {| o _ {i} |} \right. \left\{\min  \left[ \frac {\pi_ {\theta} \left(o _ {i , t} | q , o _ {i , <   t}\right)}{\pi_ {\theta_ {o l d}} \left(o _ {i , t} | q , o _ {i , <   t}\right)} \hat {A} _ {i, t}, \right. \right. \tag {2}
$$

$$
\left. \operatorname {c l i p} \left(\frac {\pi_ {\theta} \left(o _ {i , t} \mid q , o _ {i , <   t}\right)}{\pi_ {\theta_ {o l d}} \left(o _ {i , t} \mid q , o _ {i , <   t}\right)}, 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {i, t} \right] - \beta \mathbb {D} _ {K L} \left[ \pi_ {\theta} | | \pi_ {r e f} \right] \Bigg \} \Bigg ],
$$

where  $\pi_{\theta}$  and  $\pi_{\theta_{old}}$  are the current and old policy, and  $\epsilon$  and  $\beta$  are hyperparameters.

Reward Design and Validation. During RFT, we apply a two-stage validation scheme to evaluate model outputs: (1) format checking and (2) semantic correctness. The reward is mapped to the range  $[-1, 1]$ . If an output fails the format check (e.g., malformed structure or missing fields), a reward of  $-1$  is assigned. If the format is correct but the answer is semantically incorrect, the reward is 0. If both format and answer are correct, the reward is 1. For action spaces involving continuous goals, such as predicting a POINT target, we further define correctness by spatial accuracy: if the predicted point falls within the ground-truth bounding box, a reward of 1 is assigned; otherwise, 0. This fine-grained reward design encourages both syntactic correctness and task-specific accuracy.

# 3.5.2 System Optimization

Our training system adopts an asynchronous architecture that decouples rollout execution from policy updates. Once a task ID is dispatched from the global task queue, it is sampled  $n$  times according to the GRPO algorithm to generate multiple candidate responses per policy. After inference and reward computation for each sample are complete, the main process computes the advantage for the samples using GRPO's variance-reduced estimator. These advantage values are then sent to the node-level main process for policy updating. The global main process collects all necessary statistics and, when synchronization conditions are met, coordinates a unified policy update across nodes. This design ensures tight integration of GRPO's optimization logic within our distributed, asynchronous training framework.

Asynchronous Rollout. In our design, each GPU group performs inference independently and asynchronously. The inference results are first synchronized to the local node's main process. Then, each local main process communicates its inference status with a global main process, which tracks global rollout progress and coordinates training updates. During inference, each GPU group also asynchronously requests the next batch of data required for computing policy gradients. The global main process monitors the overall rollout status and, once a pre-defined synchronization condition is met, broadcasts a signal to all GPU groups to pause rollout and perform a synchronized model update. This asynchronous rollout scheme ensures that GPU groups operate efficiently without waiting for each other, thus fully utilizing computational resources.

Hierarchical Load Balancing. The asynchronous design introduces challenges related to load imbalance, particularly at two levels: intra-node (between GPU groups) and inter-node (between different compute nodes). Intra-node imbalance is addressed by constructing a global task queue from which inference tasks are dynamically dispatched to GPU groups. This design makes each GPU group consistently have access to available tasks, thereby minimizing idle time. However, nodes with differing hardware configurations or system loads can result in inter-node imbalance: some nodes may accumulate more rollout results than others. To address this, we implement a work stealing mechanism: underutilized nodes can request inference results from overburdened peers. This approach is particularly suited for large-scale, multi-modal inference outputs, which are often expensive to transmit and manage. Work stealing provides a flexible and scalable solution that avoids the drawbacks of forced synchronization across machines.

# 4 Experiments

# 4.1 GUI Grounding Capability

We evaluate GUI grounding on CAGUI through three tasks designed to assess different aspects of visual-language alignment and understanding: 1) Fun2Point. Given a description of a component's function in the GUI (e.g., "this button opens the website"), the model must locate the correct coordinates of the mentioned component; 2) Text2Point. The model is required to locate a given textual string appearing within the GUI; 3) Bbox2Text. The model receives a bounding box location on the GUI and must accurately output the corresponding textual content. Representative examples of these tasks are included in Appendix C.1.

All three grounding tasks are evaluated on the CAGUI benchmark, which was specifically curated for assessing GUI grounding capability in Chinese Android apps. The raw dataset consists of screenshots paired with corresponding XML metadata collected from real-world apps. Each XML file provides fine-grained annotations for GUI widgets, including bounding box coordinates, textual content, and component types. For the Text2Point and Bbox2Text tasks, annotations were directly extracted from the XML metadata by aligning textual content with their corresponding bounding boxes. For Fun2Point, additional function-level labels were constructed to reflect the semantic roles of GUI widgets. To generate these labels, we first overlaid bounding boxes onto the screenshots to explicitly highlight the spatial boundaries of each widget. Then, we prompted a strong VLM Qwen2.5-VL-72B to produce concise functional descriptions, yielding high-quality semantic labels for each widget.

Table 2: GUI grounding accuracy on the CAGUI benchmark over the Fun2Point, Text2Point, and Bbox2Text sub-tasks. Bold and underline indicate the best and second-best results.  

<table><tr><td>Models</td><td>Fun2Point</td><td>Text2Point</td><td>Bbox2Text</td><td>Average</td></tr><tr><td colspan="5">Closed-source Models</td></tr><tr><td>GPT-4o (Hurst et al., 2024)</td><td>22.1</td><td>19.9</td><td>14.3</td><td>18.8</td></tr><tr><td>GPT-4o with grounding (Lu et al., 2024b)</td><td>44.3</td><td>44.0</td><td>14.3</td><td>34.2</td></tr><tr><td colspan="5">Open-source Models</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2023)</td><td>59.8</td><td>59.3</td><td>50.0</td><td>56.4</td></tr><tr><td>InternVL2.5-8B (Dong et al., 2024)</td><td>17.2</td><td>24.2</td><td>45.9</td><td>29.1</td></tr><tr><td>InternVL2.5-26B (Dong et al., 2024)</td><td>14.8</td><td>16.6</td><td>36.3</td><td>22.6</td></tr><tr><td>OS-Genesis-7B (Sun et al., 2024)</td><td>8.3</td><td>5.8</td><td>4.0</td><td>6.0</td></tr><tr><td>UI-TARS-7B (Qin et al., 2025b)</td><td>56.8</td><td>66.7</td><td>1.4</td><td>41.6</td></tr><tr><td>OS-Altas-7B (Wu et al., 2025)</td><td>53.6</td><td>60.7</td><td>0.4</td><td>38.2</td></tr><tr><td>Aguvis-7B (Xu et al., 2024)</td><td>60.8</td><td>76.5</td><td>0.2</td><td>45.8</td></tr><tr><td>AgentCPM-GUI</td><td>79.1</td><td>76.5</td><td>58.2</td><td>71.3</td></tr></table>

Evaluation procedures were tailored to the input-output formats of each model. InternVL models output bounding boxes, which are evaluated against the ground-truth using the Intersection-over-Union (IoU) metric, with a threshold of 0.5 indicating a successful match. GPT-4o is augmented with OmniParser (Lu et al., 2024b), which extracts layout structures and text/icon segments before the model predicts a target box index. Models including ours generate point coordinates and are assessed by comparing them with ground-truth locations under a predefined spatial tolerance.

The results are summarized in Table 2. AgentCPM-GUI significantly outperforms all baselines across all three tasks. In particular, it achieves a large performance margin in the Bbox2Text task, where most baseline models struggle-largely due to the need for precise alignment between visual regions and text content. Despite the task's difficulty, AgentCPM-GUI attains a  $58.2\%$  accuracy, while nearly all competing models score below  $5\%$ . This highlights our model's superior grounding ability, especially in mobile interface contexts where visual complexity, small text, and overlapping elements pose unique challenges.

# 4.2 Action Prediction Capability

Table 3: Step-level action prediction performance on five GUI Agent benchmarks, in terms of Type Match (TM) and Exact Match (EM). Bold and underline indicate the best and second-best results. *OS-Atlas uses different train/test splits on GUI-Odyssey benchmark and is not directly comparable.  

<table><tr><td rowspan="2">Models</td><td colspan="2">AC-Low</td><td colspan="2">AC-High</td><td colspan="2">Odyssey</td><td colspan="2">AITZ</td><td colspan="2">CAGUI</td></tr><tr><td>TM</td><td>EM</td><td>TM</td><td>EM</td><td>TM</td><td>EM</td><td>TM</td><td>EM</td><td>TM</td><td>EM</td></tr><tr><td colspan="11">Closed-source Models</td></tr><tr><td>GPT-4o (Hurst et al., 2024)</td><td>-</td><td>19.5</td><td>-</td><td>20.8</td><td>-</td><td>20.4</td><td>70.0</td><td>35.3</td><td>3.67</td><td>3.67</td></tr><tr><td>Gemini 2.0 (Deepmind, 2024)</td><td>-</td><td>28.5</td><td>-</td><td>60.2</td><td>-</td><td>3.27</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Claude (Anthropic, 2024)</td><td>-</td><td>19.4</td><td>-</td><td>12.5</td><td>60.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="11">Open-source Models</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2023)</td><td>94.1</td><td>85.0</td><td>75.1</td><td>62.9</td><td>59.5</td><td>46.3</td><td>78.4</td><td>54.6</td><td>74.2</td><td>55.2</td></tr><tr><td>UI-TARS-7B (Qin et al., 2025b)</td><td>95.2</td><td>91.8</td><td>81.6</td><td>74.4</td><td>86.1</td><td>67.9</td><td>80.4</td><td>65.8</td><td>88.6</td><td>70.3</td></tr><tr><td>OS-Genesis-7B (Sun et al., 2024)</td><td>90.7</td><td>74.2</td><td>65.9</td><td>44.4</td><td>11.7</td><td>3.63</td><td>20.0</td><td>8.45</td><td>38.1</td><td>14.5</td></tr><tr><td>OS-Atlas-7B (Wu et al., 2025)</td><td>73.0</td><td>67.3</td><td>70.4</td><td>56.5</td><td>91.8*</td><td>76.8*</td><td>74.1</td><td>58.5</td><td>81.5</td><td>55.9</td></tr><tr><td>Aguvis-7B (Xu et al., 2024)</td><td>93.9</td><td>89.4</td><td>65.6</td><td>54.2</td><td>26.7</td><td>13.5</td><td>35.7</td><td>19.0</td><td>67.4</td><td>38.2</td></tr><tr><td>OdysseyAgent (Lu et al., 2024a)</td><td>65.1</td><td>39.2</td><td>58.8</td><td>32.7</td><td>90.8</td><td>73.7</td><td>59.2</td><td>31.6</td><td>67.6</td><td>25.4</td></tr><tr><td>AgentCPM-GUI</td><td>94.4</td><td>90.2</td><td>77.7</td><td>69.2</td><td>90.9</td><td>75.0</td><td>85.7</td><td>76.4</td><td>96.9</td><td>91.3</td></tr></table>

We conduct a comprehensive evaluation of AgentCPM-GUI on representative benchmarks: AndroidControl (Li et al., 2024a), GUI-Odyssey (Lu et al., 2024a), AITZ (Zhang et al., 2024b), and CAGUI, covering diverse GUI interaction patterns across both English and Chinese environments. Each benchmark adopts two standard evaluation metrics: Type Match (TM), which checks if the predicted action type matches the ground truth, and Exact Match (EM), which additionally requires all parameters to be correctly predicted. As shown in Table 3, AgentCPM-GUI achieves state-of-the-art performance across all benchmarks. Notably, it demonstrates strong generalization in complex multi-step scenarios, such as those in GUI-Odyssey and AITZ, significantly outperforming existing models. On the CAGUI benchmark, our model achieves  $96.9\%$  TM and  $91.3\%$  EM, substantially ahead of other models, highlighting its effectiveness in Chinese-language GUI settings.

All baseline results are from our own re-implementations to ensure fair and reproducible comparisons. We closely followed each model's official instructions and prompts where available, and applied consistent input and evaluation protocols throughout. Notably, OS-Atlas uses a different train/test split on GUI-Odyssey benchmark, so its results are not directly comparable. Our evaluation code and benchmarks are publicly released to support reproducibility and future research.

Table 4: Ablation study comparing AgentCPM-GUI before and after RFT.  

<table><tr><td rowspan="2">Models</td><td colspan="2">AC-Low</td><td colspan="2">AC-High</td><td colspan="2">Odyssey</td><td colspan="2">AITZ</td><td colspan="2">CAGUI</td></tr><tr><td>TM</td><td>EM</td><td>TM</td><td>EM</td><td>TM</td><td>EM</td><td>TM</td><td>EM</td><td>TM</td><td>EM</td></tr><tr><td>AgentCPM-GUI-SFT</td><td>87.6</td><td>83.1</td><td>78.6</td><td>69.5</td><td>86.1</td><td>66.7</td><td>79.0</td><td>61.1</td><td>96.9</td><td>91.5</td></tr><tr><td>AgentCPM-GUI-RFT</td><td>94.4</td><td>90.2</td><td>77.7</td><td>69.2</td><td>90.9</td><td>75.0</td><td>85.7</td><td>76.4</td><td>96.9</td><td>91.3</td></tr></table>

# 4.3 Effects of Reinforcement Fine-tuning

To assess the contribution of RFT, we compare our model's performance before and after RFT across all benchmarks, as shown in Table 4. On challenging datasets such as AndroidControl-Low, GUI-Odyssey, and AITZ, RFT brought significant improvements, especially in exact match accuracy. This demonstrates its effectiveness in enhancing the model's ability to handle long-horizon reasoning and complex decision-making. However, on datasets like AndroidControl-High and CAGUI, the SFT-only model already performed competitively or even slightly better. This is likely because these benchmarks have sufficiently large and diverse training sets, allowing the model to encounter similar patterns during supervised training. In such cases, imitation learning alone generalizes well, and additional reinforcement may yield limited benefit.

To monitor the optimization process, we held out a small subset of the training data as a validation set and tracked the reward curves on both the training and validation sets (Figure 2). Notably, the training reward curve shows substantial fluctuations since each point reflects a single mini-batch, while validation points are averaged over the entire held-out set. Despite the variance, the training reward trends upward, indicating effective learning. The validation reward rises steadily and plateaus around 0.82, suggesting good generalization.

![](images/141fae4ce6d375063916a2ad3773c396ad58761195764634f85b425b261a1679.jpg)  
Figure 2: Reward curves on the training and validation sets of AgentCPM-GUI.

# 5 Conclusion

We present AgentCPM-GUI, a VLM-based agent designed for GUI interaction on mobile devices. Built upon MiniCPM-V, AgentCPM-GUI is trained through a three-stage pipeline to progressively acquire grounding, action, and reasoning capabilities. To support this process, we construct a high-quality Chinese Android interaction dataset and augment it with carefully selected open-source English data, enabling the agent to generalize effectively across both Chinese and English applications. In particular, our RFT stage equips the model with stronger reasoning and planning abilities, which are essential for handling complex, long-horizon GUI tasks. We also design a compact action space with an average output length of 9.7 tokens, making the model well-suited for deployment on edge devices, which is a dimension often overlooked in previous work. Extensive experiments across public benchmarks and our newly introduced CAGUI benchmark demonstrate state-of-the-art performance, especially in Chinese-language environments. To foster further research and ensure reproducibility, we release all code, evaluation data, and model checkpoint.

# References

Anthropic. Introducing computer use, a new claude 3.5 sonnet, and claude 3.5 haiku. https://www.anthropic.com/news/3-5-models-and-computer-use, 2024.  
Hao Bai, Yifei Zhou, Jiayi Pan, Mert Cemri, Alane Suhr, Sergey Levine, and Aviral Kumar. DigiRL: Training in-the-wild device-control agents with autonomous reinforcement learning. In Advances in Neural Information Processing Systems 38, 2024.  
Hao Bai, Yifei Zhou, Li Erran Li, Sergey Levine, and Aviral Kumar. Digi-Q: Learning VLM q-value functions for training device-control agents. In International Conference on Learning Representations, 2025.  
Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A frontier large vision-language model with versatile abilities. arXiv preprint, 2023.  
Yuxiang Chai, Siyuan Huang, Yazhe Niu, Han Xiao, Liang Liu, Dingyu Zhang, Peng Gao, Shuai Ren, and Hongsheng Li. AMEX: Android multi-annotation expo dataset for mobile GUI agents. arXiv preprint, 2024.  
Wei Chen and Zhiyuan Li. Octopus v2: On-device language model for super agent. arXiv preprint, 2024.  
Wei Chen, Zhiyuan Li, and Mingyuan Ma. Octopus: On-device language model for function calling of software apis. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 329-339, 2025a.  
Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, Yuan Yao, Yankai Lin, Zhiyuan Liu, and Maosong Sun. GUICourse: From general vision language models to versatile GUI agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, 2025b.  
Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. SeeClick: Harnessing GUI grounding for advanced visual GUI agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pp. 9313-9332, 2024.  
Google Deepmind. Introducing gemini 2.0: our new ai model for the agentic era. https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/, 2024.  
Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samual Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2Web: Towards a generalist agent for the web. In Advances in Neural Information Processing Systems 36, 2023.  
Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. InternLM-XComposer2-4KHD: A pioneering large vision-language model handling resolutions from 336 pixels to 4k HD. In Advances in Neural Information Processing Systems 38, 2024.  
Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for GUI agents. In The Thirteenth International Conference on Learning Representations, 2025.  
Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for GUI agents. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, pp. 14281-14290, 2024.  
Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint, 2025.  
Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang, and Enhong Chen. Understanding the planning of LLM agents: A survey. arXiv preprint, 2024.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint, 2024.  
Geunwoo Kim, Pierre Baldi, and Stephen McAleer. Language models can solve computer tasks. In Advances in Neural Information Processing Systems 36, 2023.  
Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenshot-pro: GUI grounding for professional high-resolution computer use. arXiv preprint, 2025.  
Wei Li, William W. Bishop, Alice Li, Christopher Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. On the effects of data scale on computer control agents. arXiv preprint, 2024a.  
Yanda Li, Chi Zhang, Wanqi Yang, Bin Fu, Pei Cheng, Xin Chen, Ling Chen, and Yunchao Wei. Appagent v2: Advanced agent for flexible mobile interactions. arXiv preprint, 2024b.  
Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Weixian Lei, Lijuan Wang, and Mike Zheng Shou. Showui: One vision-language-action model for GUI visual agent. arXiv preprint, 2024.  
Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, and Fei Wu. Infigui-r1: Advancing multimodal GUI agents from reactive actors to deliberative reasoners. arXiv preprint, 2025a.  
Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint, 2025b.  
Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. GUI odyssey: A comprehensive dataset for cross-app GUI navigation on mobile devices. arXiv preprint, 2024a.  
Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based GUI agent. arXiv preprint, 2024b.  
Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Han Xiao, Shuai Ren, Guanjing Xiong, and Hongsheng Li. UIR1: Enhancing action prediction of gui agents by reinforcement learning. arXiv preprint, 2025.  
Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, Xintong Li, Jing Shi, Hongjie Chen, Viet Dao Lai, Zhouhang Xie, Sungchul Kim, Ruiyi Zhang, Tong Yu, Md. Mehrab Tanjim, Nesreen K. Ahmed, Puneet Mathur, Seunghyun Yoon, Lina Yao, Branislav Kveton, Thien Huu Nguyen, Trung Bui, Tianyi Zhou, Ryan A. Rossi, and Franck Dernoncourt. Gui agents: A survey. arXiv preprint, 2024.  
OpenAI. Gpt-4v(ison) system card. Tech Report, 2023.  
OpenAI. Reinforcement fine-tuning. https://platform.openai.com/docs/guides/reinforcement-fine-tuning, 2024.  
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems 35, 2022.  
Georgios Papoudakis, Thomas Coste, Zhihao Wu, Jianye Hao, Jun Wang, and Kun Shao. Appvlm: A lightweight vision language model for online app control. arXiv preprint, 2025.  
Cheng Qian, Bingxiang He, Zhong Zhuang, Jia Deng, Yujia Qin, Xin Cong, Zhong Zhang, Jie Zhou, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Tell me more! towards implicit user intention understanding of language model driven agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pp. 1088-1113, 2024.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. Toolllm: Facilitating large language models to master 16000+ real-world apis. In The Twelfth International Conference on Learning Representations, 2024.  
Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Xuanhe Zhou, Yufei Huang, Chaojun Xiao, Chi Han, Yi R. Fung, Yusheng Su, Huadong Wang, Cheng Qian, Runchu Tian, Kunlun Zhu, Shihao Liang, Xingyu Shen, Bokai Xu, Zhen Zhang, Yining Ye, Bowen Li, Ziwei Tang, Jing Yi, Yuzhang Zhu, Zhenning Dai, Lan Yan, Xin Cong, Yaxi Lu, Weilin Zhao, Yuxiang Huang, Junxi Yan, Xu Han, Xian Sun, Dahai Li, Jason Phang, Cheng Yang, Tongshuang Wu, Heng Ji, Guoliang Li, Zhiyuan Liu, and Maosong Sun. Tool learning with foundation models. ACM Computing Surveys, 57(4): 101:1-101:40, 2025a.  
Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. UI-TARS: pioneering automated GUI interaction with native agents. arXiv preprint, 2025b.  
Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy P. Lillicrap. Android in the wild: A large-scale dataset for android device control. arXiv preprint, 2023.  
Christopher Rawles, Sarah Clinkemaaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint, 2024.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint, 2017.  
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint, 2024.  
Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, Ben Kao, Guohao Li, Junxian He, Yu Qiao, and Zhiyong Wu. Os-genesis: Automating GUI agent trajectory construction via reverse task synthesis. arXiv preprint, 2024.  
Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang.  
Reason-rft: Reinforcement fine-tuning for visual reasoning. arXiv preprint, 2025.  
Luong Quoc Trung, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. Reft: Reasoning with reinforced fine-tuning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pp. 7601-7614, 2024.  
Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint, 2024a.  
Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024b.  
Shuai Wang, Weiwen Liu, Jingxuan Chen, Weinan Gan, Xingshan Zeng, Shuai Yu, Xinlong Hao, Kun Shao, Yasheng Wang, and Ruiming Tang. GUI agents with foundation models: A comprehensive survey. arXiv preprint, 2024c.  
Taiyi Wang, Zhihao Wu, Jianheng Liu, Jianye Hao, Jun Wang, and Kun Shao. Distrl: An asynchronous distributed reinforcement learning framework for on-device control agent. In The Thirteenth International Conference on Learning Representations, 2025.  
Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, and Yu Qiao. OS-ATLAS: foundation action model for generalist GUI agents. In The Thirteenth International Conference on Learning Representations, 2025.

Xiaobo Xia and Run Luo. GUI-R1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458, 2025.  
Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous GUI interaction. arXiv preprint, 2024.  
Yuhao Yang, Yue Wang, Dongxu Li, Ziyang Luo, Bei Chen, Chao Huang, and Junnan Li. Aria-ui: Visual grounding for GUI instructions. arXiv preprint, 2024.  
Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm-v: A GPT-4V level MLLM on your phone. arXiv preprint, 2024.  
Simon Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Peter Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, and Sergey Levine. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. In Advances in Neural Information Processing Systems 38, 2024.  
Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Guyue Liu, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, and Qi Zhang. Large language model-brained GUI agents: A survey. arXiv preprint, 2024a.  
Chi Zhang, Zhao Yang, Jiaxuan Liu, Yanda Li, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, 2025.  
Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the zoo: Chain-of-action-thought for GUI agents. In Findings of the Association for Computational Linguistics, 2024b.  
Zhuosheng Zhang and Aston Zhang. You only look at screens: Multimodal chain-of-action agents. In Findings of the Association for Computational Linguistics, 2024.  
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models. arXiv preprint, 2023.  
Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v(ision) is a generalist web agent, if grounded. In *Forty-first International Conference on Machine Learning*, 2024.  
Hengguang Zhou, Xinui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1-zero's "aha moment" in visual reasoning on a 2b non-sft model. arXiv preprint, 2025.

# A Training Details

We list the main hyperparameters for the SFT and RFT stages in Table 5 and Table 6, respectively.

Table 5: Training parameters for Stage II: Supervised Fine-tuning.  

<table><tr><td>Parameter</td><td>Default Value</td><td>Description</td></tr><tr><td>model_max_length</td><td>2304</td><td>Maximum sequence length</td></tr><tr><td>max_line_res</td><td>1120</td><td>Maximum image resolution for the longest axis</td></tr><tr><td>per_device_train_batch_size</td><td>1</td><td>Training batch size per device</td></tr><tr><td>gradient Accumulation_steps</td><td>1</td><td>Gradient accumulation steps</td></tr><tr><td>num_train_epochs</td><td>3</td><td>Number of training epochs</td></tr><tr><td>learning_rate</td><td>1e-5</td><td>Learning rate</td></tr><tr><td>weight Decay</td><td>0.1</td><td>Weight decay coefficient</td></tr><tr><td>adam_beta1</td><td>0.9</td><td>Adam optimizer beta1 parameter</td></tr><tr><td>adam_beta2</td><td>0.999</td><td>Adam optimizer beta2 parameter</td></tr><tr><td>max_grad_norm</td><td>N/A</td><td>Gradient clipping disabled</td></tr><tr><td>lr_scheduler_type</td><td>cosine</td><td>Learning rate scheduler type</td></tr><tr><td>warmup_ratio</td><td>0.05</td><td>Linear warmup ratio</td></tr><tr><td>bf16</td><td>True</td><td>Use bfloat16 precision</td></tr><tr><td>gradient_checkpointing</td><td>False</td><td>Whether using gradient checkpointing</td></tr><tr><td>deepspeed</td><td>ZeRO-2</td><td>Deepspeed optimization stage</td></tr></table>

Table 6: Training parameters for Stage III: Reinforcement Fine-tuning.  

<table><tr><td>Parameter</td><td>Default Value</td><td>Description</td></tr><tr><td>max_prompt_length</td><td>16384</td><td>Maximum prompt length</td></tr><tr><td>max Completion_length</td><td>512</td><td>Maximum completion length</td></tr><tr><td>max_line_res</td><td>1120</td><td>Maximum image resolution for the longest axis</td></tr><tr><td>num GENERATIONS</td><td>8</td><td>Number of generations</td></tr><tr><td>per_device_train_batch_size</td><td>1</td><td>Training batch size per device</td></tr><tr><td>gradient Accumulation_steps</td><td>32</td><td>Gradient accumulation steps</td></tr><tr><td>learning_rate</td><td>1e-6</td><td>Learning rate</td></tr><tr><td>num_train_epochs</td><td>3</td><td>Number of training epochs</td></tr><tr><td>weight Decay</td><td>0.1</td><td>Weight decay coefficient</td></tr><tr><td>adam_beta2</td><td>0.99</td><td>Adam optimizer beta2 parameter</td></tr><tr><td>max_grad_norm</td><td>1.0</td><td>Maximum gradient norm for clipping</td></tr><tr><td>lr_scheduler_type</td><td>cosine</td><td>Learning rate scheduler type</td></tr><tr><td>beta</td><td>0.04</td><td>KL divergence coefficient</td></tr><tr><td>bf16</td><td>True</td><td>Use bfloat16 precision</td></tr></table>

# B Evaluation Details

To ensure fair and consistent evaluation across all models, we adopt a unified evaluation framework. Since different models may define their own action formats and conventions, their outputs are first converted into a shared action representation defined by AgentCPM-GUI. This normalization allows us to compare models under the same evaluation criteria and metrics. In the following, we provide representative input prompts for each model, detail the evaluation settings and hyperparameters, and describe how action space conversion is performed when applicable.

# B.1 Qwen2.5-VL-7B

# B.1.1 Data example

```txt
Qwen2.5-VL-7B Data Example   
System Message   
You are a helpful assistant. # Tools You may call one or more functions to assist with the user query. You are provided with function signatures within</tools>XML tags: <tools> {"type": "function", "function": {"name_for_human": "mobile\use", "name": "mobile\use", " description": "Use a touchscreen to interact with a mobile device, and take screenshots. \* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc. \* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. \* The screen's resolution is 1092x2408. \* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {" description": "The action to perform. The available actions are: \* `key': Perform a key event on the mobile device. - This supports adb's `keyevent` syntax. - Examples: "\\"volume\up\"", "\\"volume\down\"", "\\"power\"", "\\"camera\"", "\\"clear\"". \*`click': Click the point on the screen with coordinate (x, y). \*`long\press': Press the point on the screen with coordinate (x, y) for specified seconds. \*`swifte': Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 ( x2, y2). \*`type': Input the specified text into the activated input box. \*`system\button': Press the system button. \*`open': Open an app on the device. \*`wait': Wait specified seconds for the change to happen. \*`terminate': Terminate the current task and report its completion status.", "enum": ["key", "click", "long\press", "swifte", "type", "system\button", "open", "wait", "terminate"], "type": "string ", "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click', `action= long\press', and `action=swifte'","type": "array" }, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swifte'","type": "array" }, "text": {"description": "Required only by `action=key', `action=type', and `action=open'","type": "string" }, "time": {"description": "The seconds to wait. Required only by `action=long\press' and 'action=
```

```txt
wait\","type": "number"},{"button": {"description":"Back means returning to the previous interface,Home means returning to the desktop,Menu means opening the application background menu,and Enter means pressing the enter. Required only by \`action=system\\ _button\\","enum": ["Back","Home","Menu","Enter"], "type": "string"},{status": {" description":"The status of the task. Required only by \`action=terminate\","type":"string","enum": ["success","failure"]}},"required": ["action"], "type": "object"], "args\_\format": " Format the arguments as a JSON object."}} </tools> For each function call,return a json object with function name and arguments within</ tool\call>< tool\call>XML tags: <tool_call> {"name": <function-name>, "arguments": <args-json-object>} </tool_call>   
User   
The user query: [user_request] Current step query: low_lew Instruction (included only when low_lewinstruction is defined) Task progress (You have done the following operation on the current device): [history_actions] [current Screenshots]   
Assistant   
[thought_and_action]
```

# B.1.2 Action Space Mapping

Table 7 shows the action space mapping from Qwen2.5-VL-7B to the standardized representation. Two key differences must be addressed during conversion. First, Qwen2.5-VL-7B expresses duration in seconds for actions such as long_press and wait, whereas AgentCPM-GUI expects time in milliseconds. Second, Qwen2.5-VL-7B produces absolute screen coordinates (in pixels) for spatial actions like click, long_press, and swipe, while AgentCPM-GUI uses normalized coordinates in the range [0, 1000] relative to screen size.

Table 7: Action space mapping from Qwen2.5-VL-7B to AgentCPM-GUI.  

<table><tr><td>Qwen2.5-VL-7B</td><td>Input Parameters</td><td>AgentCPM-GUI</td></tr><tr><td>click</td><td>coordinate = (x, y)</td><td>{&quot;POINT&quot;: [int(x/width*1000), int(y/height*1000)]}</td></tr><tr><td>long_press</td><td>coordinate = (x, y), time</td><td>{&quot;POINT&quot;: [x, y], &quot;duration&quot;: time*1000}</td></tr><tr><td>swipe</td><td>coordinate = (x1, y1), coordinate2 = (x2, y2)</td><td>{&quot;POINT&quot;: [x1, y1], &quot;to&quot;: direction}Note: direction is computed from two points</td></tr><tr><td>type</td><td>text</td><td>{&quot;TYPE&quot;: text}</td></tr><tr><td>system_button</td><td>button = Back / Home / Enter</td><td>{&quot;PRESS&quot;: BACK/HOME/ENTER}</td></tr><tr><td>terminate</td><td>None</td><td>{&quot;STATUS&quot;: &quot;finish&quot;}</td></tr><tr><td>wait</td><td>time</td><td>{&quot;duration&quot;: time*1000}</td></tr></table>

# B.1.3 Hyperparameters

We adopt the same hyperparameter settings as used in Qwen2.5-VL-7B for fair comparison, as summarized in Table 8.

Table 8: Inference hyperparameters for Qwen2.5-VL-7B.  

<table><tr><td>Parameter</td><td>Default Value</td><td>Description</td></tr><tr><td>do_sample</td><td>True</td><td>Whether to use sampling (replaces greedy)</td></tr><tr><td>top_p</td><td>0.01</td><td>Nucleus sampling threshold</td></tr><tr><td>top_k</td><td>1</td><td>Top-k sampling limit</td></tr><tr><td>temperature</td><td>0.01</td><td>Controls sampling randomness</td></tr><tr><td>repetition_penalty</td><td>1.0</td><td>Penalty factor for repetition</td></tr><tr><td>max_new_tokens</td><td>2048</td><td>Maximum number of new tokens to generate</td></tr></table>

# B.2 UI-TARS

# B.2.1 Data example

<table><tr><td>UI-TARS Data Example</td></tr><tr><td></td></tr><tr><td>System Message</td></tr><tr><td>You are a helpful assistant.</td></tr><tr><td>User</td></tr><tr><td>You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.</td></tr><tr><td>Output Format</td></tr><tr><td>Thought: . . .Action: . . .</td></tr><tr><td>Action Space</td></tr><tr><td>click(start_box=''&lt;box_start|(x1,y1)&lt;box_end&gt;')long_press(start_box=''&lt;box_start|(x1,y1)&lt;box_end&gt;',' time="")type(content="”)scroll的方向='down or up or right or left')press_back()press_home()wait()finished() # Submit the task regardless of whether it succeeds or fails.</td></tr><tr><td>Note</td></tr><tr><td>- Use English in Thought part.- Summarize your next action (with its target element) in one sentence in Thought part.</td></tr><tr><td>User Instruction[user_request]</td></tr><tr><td>User</td></tr><tr><td>[history_scrrape]</td></tr><tr><td>Assistant</td></tr><tr><td>[history_thought_and_action]</td></tr><tr><td>User</td></tr><tr><td>[current_scrrape]</td></tr><tr><td>Assistant(included only when low_lewInstruction is defined)</td></tr><tr><td>Thought: [low_lewInstruction]
Action:</td></tr><tr><td>Assistant</td></tr><tr><td>[thought_and_action]</td></tr></table>

# B.2.2 Action Space Mapping

Table 9 shows the action space mapping from UI-TARS to the standardized representation. Since UI-TARS and AgentCPM-GUI define scroll directions oppositely, the direction must be reversed during conversion.

Table 9: Action space mapping from UI-TARS to AgentCPM-GUI.  

<table><tr><td>UI-TARS</td><td>Input Format</td><td>AgentCPM-GUI</td></tr><tr><td>click(...)</td><td>start_box with (x,y)</td><td>{&quot;POINT&quot;:[x,y]}</td></tr><tr><td>long_press(...)</td><td>start_box with (x,y), time=&#x27;ms&#x27; (optional)</td><td>{&quot;POINT&quot;:[x,y],&quot;duration&quot;:time(default 1000)}</td></tr><tr><td>type(...)</td><td>content=&#x27;text&#x27;</td><td>{&quot;TYPE&quot;:text}</td></tr><tr><td>scroll(...)</td><td>direction=&#x27;up/down/left/right&#x27;</td><td>{&quot;POINT&quot;:[500,500],&quot;to&quot;:reversed direction&quot;}Note: direction is reversed (e.g., up → down)</td></tr><tr><td>press_back()</td><td>-</td><td>{&quot;PRESS&quot;:BACK}</td></tr><tr><td>press_home()</td><td>-</td><td>{&quot;PRESS&quot;:HOME}</td></tr><tr><td>wait()</td><td>-</td><td>{&quot;duration&quot;:200}</td></tr><tr><td>finished()</td><td>-</td><td>{&quot;STATUS&quot;:&quot;finish&quot;}</td></tr></table>

# B.3 OS-ATLAS

# B.3.1 Data example

<table><tr><td>OS-ATLAS Data Example</td></tr><tr><td></td></tr><tr><td>System Message</td></tr><tr><td>You are a helpful assistant.</td></tr><tr><td>User</td></tr><tr><td></td></tr></table>

You are a foundational action model capable of automating tasks across various digital environments, including desktop systems like Windows, macOS, and Linux, as well as mobile platforms such as Android and iOS. You also excel in web browser environments. You will interact with digital devices in a human-like manner: by reading screenshots, analyzing them, and taking appropriate actions.

Your expertise covers two types of digital tasks:

- Grounding: Given a screenshot and a description, you assist users in locating elements mentioned. Sometimes, you must infer which elements best fit the description when they aren't explicitly stated.  
- Executable Language Grounding: With a screenshot and task instruction, your goal is to determine the executable actions needed to complete the task.

You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

# 1. Basic Actions

Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability.

Basic Action 1: CLICK

- purpose: Click at the specified position.  
-format:CLICK[x-axis,y-axis]]</point>  
- example usage: CLICK <point>[[101, 872]]</point>

# Basic Action 2: TYPE

- purpose: Enter specified text at the designated location.  
-format: TYPE [input text]  
- example usage: TYPE [Shanghai shopping mall]

# Basic Action 3: SCROLL

- purpose: Scroll in the specified direction.  
-format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]  
- example usage: SCROLL [UP]

# 2.Custom Actions

Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

# Custom Action 1: LONG_press

- purpose: Long press at the specified position.  
- format: LONG_press <point>[[x-axis, y-axis]]</point>  
- example usage: LONG_press <point>[[101, 872]]</point>

# Custom Action 2: PRESS_BACK

- purpose: Press a back button to navigate to the previous screen.  
- format: PRESS_BACK  
- example usage: PRESS_BACK

# Custom Action 3: PRESS_HOME

- purpose: Press a home button to navigate to the home page.  
- format: PRESS_HOME  
- example usage: PRESS_HOME

# Custom Action 4: PRESS_RECENT

- purpose: Press the recent button to view or switch between recently used applications.  
- format: PRESS_RECENT

- example usage: PRESS_RECENT

Custom Action 5: WAIT

- purpose: Wait for the screen to load.  
- format: WAIT  
- example usage: WAIT

Custom Action 6: COMPLETE

- purpose: Indicate the task is finished.  
- format: COMPLETE  
- example usage: COMPLETE

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.

Thoughts: Clearly outline your reasoning process for current step.

Actions: Specify the actual actions you will take based on your reasoning.

Your current task instruction, action history, and associated screenshot are as follows:

Screenshot:[current Screenshots]

Task: [user_request] You need to: [low_lew Instruction](included only when low_lewInstruction is defined)

History:

[history_low_lewInstruction](included only when low_lew Instruction is defined)

Assistant

[thought_and_action]

# B.3.2 Action Space Mapping

Table 10 shows the action space mapping from OS-ATLAS to the standardized representation. When evaluating the AndroidControl-Low setting, we found that the model's predicted scroll direction is often opposite to that indicated in the low-level instruction. Therefore, the scroll direction is reversed during evaluation.

Table 10: Action space mapping from OS-Atlas to AgentCPM-GUI.  

<table><tr><td>OS-Atlas</td><td>Input Format</td><td>AgentCPM-GUI</td></tr><tr><td>CLICK</td><td>[[x, y]]</td><td>{&quot;POINT&quot;:[x, y]}</td></tr><tr><td>LONG_press</td><td>[[x, y]]</td><td>{&quot;POINT&quot;:[x, y], &quot;duration&quot;:1000}</td></tr><tr><td>TYPE</td><td>{text]</td><td>{&quot;TYPE&quot;:text}</td></tr><tr><td>SCROLL</td><td>[direction]</td><td>{&quot;POINT&quot;:[500, 500], &quot;to&quot;:direction}Note: if use_low Instruction is True, direction is reversed: up←down, left←right</td></tr><tr><td>PRESS_BACK</td><td>-</td><td>{&quot;PRESS&quot;:BACK}</td></tr><tr><td>PRESS_HOME</td><td>-</td><td>{&quot;PRESS&quot;:HOME}</td></tr><tr><td>PRESS_RECENT</td><td>-</td><td>{&quot;PRESS&quot;:RECENT}</td></tr><tr><td>WAIT</td><td>-</td><td>{&quot;duration&quot;:200}</td></tr><tr><td>COMPLETE</td><td>-</td><td>{&quot;STATUS&quot;:&quot;finish&quot;}</td></tr></table>

# B.4 OS-Genesis

# B.4.1 Data Example

For the GUI-Odyssey, AITZ, and CAGUI benchmarks, we construct evaluation prompts following the format described in Data Example. For AndroidControl, we adopt the official evaluation code provided in the benchmark's GitHub repository.

<table><tr><td>OS-Genesis Data Example</td></tr><tr><td></td></tr><tr><td>System Message</td></tr><tr><td>You are a helpful assistant.</td></tr><tr><td>User</td></tr><tr><td>You are a GUI task expert, I will provide you with a high-level instruction, an action history, a screenshot with its corresponding accessibility tree.</td></tr><tr><td>High-level instruction: [user_request]</td></tr><tr><td>Action history:</td></tr><tr><td>Accessibility tree:</td></tr><tr><td>Please generate the low-level thought and action for the next step.</td></tr><tr><td>Assistant</td></tr><tr><td>[thought_and_action]</td></tr></table>

# B.4.2 Action Space Mapping

Table 11 shows the action space mapping from OS-Genesis to the standardized representation. Similar to OS-ATLAS, the predicted scroll direction on AndroidControl-Low is often opposite to the instruction, and is therefore reversed during evaluation.

Table 11: Action space mapping from OS-Genesis to AgentCPM-GUI.  

<table><tr><td>OS-Genesis</td><td>Input Fields</td><td>AgentCPM-GUI</td></tr><tr><td>type</td><td>text</td><td>{&quot;TYPE&quot;:text}</td></tr><tr><td>click</td><td>x,y</td><td>{&quot;POINT&quot;:[x,y]}</td></tr><tr><td>long_press</td><td>x,y</td><td>{&quot;POINT&quot;:[x,y],&quot;duration&quot;:1000}</td></tr><tr><td>dismiss</td><td>x,y</td><td>{&quot;POINT&quot;:[x,y]}</td></tr><tr><td>get_text</td><td>x,y</td><td>{&quot;POINT&quot;:[x,y]}</td></tr><tr><td>navigate_home</td><td>-</td><td>{&quot;PRESS&quot;:HOME}</td></tr><tr><td>navigate_back</td><td>-</td><td>{&quot;PRESS&quot;:BACK}</td></tr><tr><td rowspan="2">scroll</td><td rowspan="2">direction</td><td>{&quot;POINT&quot;:[500,500],&quot;to&quot;:direction}</td></tr><tr><td>Note: If use_low Instruction is True, direction is reversed: up←down, left←right</td></tr><tr><td>wait</td><td>-</td><td>{&quot;duration&quot;:200}</td></tr></table>

# B.5 OdysseyAgent

# B.5.1 Data example

Following the official implementation, OdysseyAgent's input consists of the current instruction along with a history of images and their associated actions.

<table><tr><td>OdysseyAgent Data Example</td></tr><tr><td></td></tr><tr><td>System Message</td></tr><tr><td>You are a helpful assistant.</td></tr><tr><td>User</td></tr><tr><td>Picture 1: &lt;img&gt;image_path&lt;/img&gt; 
I&#x27;m looking for guidance on how to [instruction] 
Previous screenshots: &lt;img&gt;image-history: image_path&lt;/img&gt; 
Previous Actions: 1. [Action 1] 
2. [Action 2]. 
…</td></tr><tr><td>Assistant</td></tr><tr><td>[Action]</td></tr></table>

# B.5.2 Action Space Mapping

Table 12 shows the action space mapping from OdysseyAgent to the standardized representation. The output format of OdysseyAgent is largely compatible with AgentCPM-GUI. The only exception is the RECENT action, which is not part of the AgentCPM-GUI action space and is therefore ignored during evaluation.

Table 12: Action space mapping from OdysseyAgent to AgentCPM-GUI.  

<table><tr><td>OdysseyAgent</td><td>Input Fields</td><td>AgentCPM-GUI</td></tr><tr><td>CLICK</td><td>x, y</td><td>{&quot;POINT&quot;:[x,y]}</td></tr><tr><td>LONG_press</td><td>x, y</td><td>{&quot;POINT&quot;:[x,y],&quot;duration&quot;:1000}</td></tr><tr><td>SCROLL</td><td>direction</td><td>{&quot;POINT&quot;:[500,500],&quot;to&quot;:direction}</td></tr><tr><td>TYPE</td><td>text</td><td>{&quot;TYPE&quot;:text}</td></tr><tr><td>HOME</td><td>-</td><td>{&quot;PRESS&quot;:HOME}</td></tr><tr><td>BACK</td><td>-</td><td>{&quot;PRESS&quot;:BACK}</td></tr><tr><td>COMPLETE</td><td>-</td><td>{&quot;STATUS&quot;:&quot;finish&quot;}</td></tr><tr><td>IMPOSSIBLE</td><td>-</td><td>{&quot;STATUS&quot;:&quot;impossible&quot;}</td></tr></table>

# B.5.3 Hyperparameters

We follow the original implementation for inference, enabling the image_history option to incorporate temporal context. Specifically, we store the last 4 actions and their corresponding images. The inference is conducted with the torch seed set to 1234 and the random seed set to 2020 to ensure reproducibility.

# B.6 Aguvis-7B

# B.6.1 Data Example

<table><tr><td>Aguvis Data Example</td></tr><tr><td></td></tr><tr><td>System Message</td></tr><tr><td>You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.
You have access to the following functions:
- {&quot;name&quot;: &quot;mobile.swipe&quot;, &quot;description&quot;: &quot;Swipe on the screen&quot;, &quot;parameters&quot;: {&quot;type&quot;: &quot;object&quot;,
&quot;properties&quot;: {&quot;from_coord&quot;: {&quot;type&quot;: &quot;array&quot;, &quot;items&quot;: {&quot;type&quot;: &quot;number&quot;},
&quot;description&quot;: &quot;The starting coordinates of the swipe&quot;}, &quot;to_coord&quot;: {&quot;type&quot;: &quot;array&quot;, &quot;items&quot;: {&quot;type&quot;: &quot;number&quot;},
&quot;description&quot;: &quot;The ending coordinates of the swipe&quot;}}, &quot;required&quot;: [&quot;from_coord&quot;, &quot;to_coord&quot;]
- {&quot;name&quot;: &quot;mobile.home&quot;, &quot;description&quot;: &quot;Press the home button&quot;}
- {&quot;name&quot;: &quot;mobile.back&quot;, &quot;description&quot;: &quot;Press the back button&quot;}
- {&quot;name&quot;: &quot;mobile.wait&quot;, &quot;description&quot;: &quot;wait for the change to happen&quot;, &quot;parameters&quot;: {&quot;type&quot;: &quot;object&quot;, &quot;properties&quot;: {&quot;seconds&quot;: {&quot;type&quot;: &quot;number&quot;, &quot;description&quot;: &quot;The seconds to wait&quot;},
&quot;required&quot;: [&quot;seconds&quot;]}])
- {&quot;name&quot;: &quot;mobile.long_press&quot;, &quot;description&quot;: &quot;Long press on the screen&quot;, &quot;parameters&quot;: {&quot;type&quot;: &quot;object&quot;, &quot;properties&quot;: {&quot;x&quot;: {&quot;type&quot;: &quot;number&quot;, &quot;description&quot;: &quot;The x coordinate of the long press&quot;},
&#x27;y&#x27;: {&quot;type&quot;: &quot;number&quot;, &quot;description&quot;: &quot;The y coordinate of the long press&quot;},
&quot;required&quot;: [&quot;x&quot;, &quot;y&quot;]}-
- {&quot;name&quot;: &quot;mobile.open_app&quot;, &quot;description&quot;: &quot;Open an app on the device&quot;, &quot;parameters&quot;: {&quot;type&quot;: &quot;object&quot;, &quot;properties&quot;: {&quot;app_name&quot;: {&quot;type&quot;: &quot;string&quot;, &quot;description&quot;: &quot;The name of the app to open&quot;}}
,&quot;required&quot;: [&quot;app_name&quot;]}</td></tr><tr><td>User</td></tr><tr><td>Please generate the next move according to the ui screenshot, instruction and previous actions.
Instruction: [Instruction]
Previous actions: [previous ACTIONS]</td></tr><tr><td>Assistant</td></tr><tr><td>[thought and Action]</td></tr></table>

Table 13: Action space mapping from Aguvis to AgentCPM-GUI.  

<table><tr><td>Aguvis</td><td>Input Fields</td><td>AgentCPM-GUI</td></tr><tr><td>pyautogui.click</td><td>x, y</td><td>{&quot;POINT&quot;:[x*1000,y*1000]}</td></tr><tr><td>mobile.long_press</td><td>x, y</td><td>{&quot;POINT&quot;:[x*1000,y*1000],&quot;duration&quot;:1000}</td></tr><tr><td>pyautogui scroll(/hscroll(   )</td><td>direction</td><td>{&quot;POINT&quot;:[500,500],&quot;to&quot;: direction]Note: scroll performs vertical, and hscrollperforms horizontal swipes</td></tr><tr><td>pyautogui.write</td><td>text</td><td>{&quot;TYPE&quot;:text}</td></tr><tr><td>mobile.home(/</td><td>-</td><td>{&quot;PRESS&quot;:HOME}</td></tr><tr><td>mobile.back(   )</td><td>-</td><td>{&quot;PRESS&quot;:BACK}</td></tr><tr><td>mobile terminat(   )</td><td>-</td><td>{&quot;STATUS&quot;:&quot;finish&quot;}</td></tr><tr><td>mobile.open_app</td><td>app_name</td><td>-</td></tr><tr><td>mobile.wait</td><td>[time]</td><td>{&quot;duration&quot;:3000}</td></tr></table>

# B.6.2 Action Space Mapping

Table 13 shows the action space mapping from Aguvis to the standardized representation. All coordinates in Aguvis are in the range [0, 1] and are scaled accordingly during conversion. Swipe actions are mapped following the definition in the pyautogui package. Since AgentCPM-GUI does not include an "open app" action, it is ignored during evaluation.

# B.6.3 Hyperparameters

The hyper parameters are the same as the origin implementation. To be specific, we choose "self-plan" mode during inference, with temperature set as 0 and generate only 1024 new max tokens. Historical actions are not included during inference, as their inclusion leads to abnormal model behavior.

# C CAGUI Benchmark

# C.1 CAGUI_Grounding

We provide examples from the three tasks that constitute the grounding benchmark, each containing 1,500 samples. The Text2Bbox and Bbox2Text tasks are based on the same dataset. Each bounding box is defined by four absolute coordinates in the format  $< x_{\min}, y_{\min}, x_{\max}, y_{\max} >$ , with the origin located at the top-left corner of the screen.

<table><tr><td>Text2Point Data Examples</td></tr><tr><td></td></tr><tr><td>Text</td></tr><tr><td>QQ音乐</td></tr><tr><td>Bounding Box</td></tr><tr><td>&lt;643, 462, 849, 744&gt;</td></tr><tr><td>Prompt of AgentCPM-GUI</td></tr><tr><td>你是一个GUI组件定位的专家，擅长输出图片上文本对应的坐标。你的任务是根据给定的GUI截图和图中某个文本输出该文本的坐标。输入：屏幕截图，文本描述输出：文本的相对坐标的中心点,POINT:[......]为格式</td></tr></table>

<table><tr><td>Bbox2Text Data Examples</td></tr><tr><td></td></tr><tr><td>Bounding Box</td></tr><tr><td>&lt;60, 120, 132, 192&gt;</td></tr><tr><td>Bounding Box</td></tr><tr><td>返回</td></tr><tr><td>Prompt of AgentCPM-GUI</td></tr><tr><td>你是一个GUI组件文字识别的专家，擅长根据组件的边界框（bounding box）描述输出对应的文字。你的任务是根据给定的GUI截图和图中某个组件的边界框输出组件的中的文字。输入：屏幕截图，边界框的坐标输出：组件中的文本</td></tr><tr><td>Fun2Point Data Examples</td></tr><tr><td></td></tr><tr><td>Function</td></tr><tr><td>UI元素是一个菜单按钮。其主要功能是弹出一个菜单面板，允许用户选择不同的功能选项。通常可以通过点击该按钮触发，点击后会展示一个下拉或侧滑菜单，用户可以在其中进行进一步操作，例如切换功能页面或设置选项。</td></tr><tr><td>Bounding Box</td></tr><tr><td>&lt;1061, 2424, 1159, 2522&gt;</td></tr><tr><td>Prompt of AgentCPM-GUI</td></tr><tr><td>你是一个GUI组件定位的专家，擅长根据组件的功能描述输出对应的坐标。你的下一步操作是根据给定的GUI截图和图中某个组件的功能描述点击组件的中心位置。坐标为相对于屏幕左上角位原点的相对位置，并且按照宽高比例缩放到0~1000输入：屏幕截图，功能描述输出：点击操作，以POINT: [......]为格式，其中不能存在任何非坐标字符</td></tr></table>

# C.2 CAGUI_Agent

We present examples of our dataset tasks, each consisting of a query, a screenshot, and the corresponding answer operation. The system prompt used to evaluate AgentCPM-GUI is also included. In total, the benchmark comprises 600 tasks, which together contain 4,516 single-step images. During evaluation, inputs to AgentCPM-GUI follow the standard chat format. Each user message contains both the task query and the associated screenshot, structured as a list with two elements: a text string formatted as "<Question>{query}</Question>\n当前屏幕截图：" and the corresponding image.

<table><tr><td>Agent Data Examples</td></tr><tr><td></td></tr><tr><td>Query</td></tr><tr><td>请优酷视频根据我的历史记录播放7天内观看超过60%的短视频。</td></tr><tr><td>Operation</td></tr><tr><td>Action Type: Click
Action Detail: [0.13, 0.61]</td></tr><tr><td>System Prompt of AgentCPM-GUI</td></tr><tr><td># Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。</td></tr><tr><td># Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。</td></tr><tr><td># Rule
-以紧凑JSON格式输出
-输出操作必须遵循Schema约束</td></tr><tr><td># Schema
{
    &quot;type&quot;: &quot;object&quot;,
    &quot;description&quot;: &quot;执行操作并决定当前任务状态&quot;,
    &quot;additionalProperties&quot;: false,
    &quot;properties&quot;: {</td></tr></table>

```txt
"thought": { "type": "string", "description": "智能体的思维过程" }, "POINT": { "$ref": "#/$refs/Location", "description": "点击屏幕上的指定位置" }, "to": { "description": "移动,组合手势参数", "oneOf": [ {"enum": [ "up", "down", "left", "right"], "description": "从当前点(POINT)出发,执行滑动手势操作,方向包括向上、向下、向左、向右"} }, {"ref": "#/$refs/Location", "description": "移动到某个位置"} ], }, "duration": { "type": "integer", "description": "动作执行的时间或等待时间,毫秒", "minimum": 0, "default": 200 }, "PRESS": { "type": "string", "description": "触发特殊按键,HOME为回到主页按钮,BACK为返回按钮,ENTER为回车按钮", "enum": [ "HOME", "BACK", "ENTER"] }, "TYPE": { "type": "string", "description": "输入文本" }, "STATUS": { "type": "string", "description": "当前任务的状态。特殊情况:satisfied,无需操作; impossible,任务无法完成;interrupt,任务中断;need_feedback,需要用户反馈"; "enum": [ "continue", "finish", "satisfied", "impossible", interrupt",
```

```json
"need_feedback"
],
"default": "continue"
}
},
"\\(refs": \{
	"Location": \{
		"type": "array",
		"description": "坐标为相对于屏幕左上角位原点的相对位置,并且按照宽高比例缩放
到0~1000,数组第一个元素为横坐标x,第二个元素为纵坐标y",
		"items": \{
			"type": "integer",
			"minimum": 0,
			"maximum": 1000
	\},
		"minItems": 2,
		"maxItems": 2
\}
\}
```

# D Case Study

We demonstrate GUI agent tasks on a real Xiaomi 12S device running MIUI 14.0.11. All interactions with the graphical interface are carried out via ADB control based on the AgentCPM-GUI's predicted actions. The original input and output were in Chinese and translated into English.

![](images/1579d132efe47ba078c4c77809cbc1ea0559fe670a1a5ab3f37318a3844a7762.jpg)

User Task: Go to Bilibili to watch Li Ziqi's latest video and give it a like.

![](images/9628227f57e07983c6b2fc36aac56e48af872d4a55281e38415abcfdd91b270c.jpg)

![](images/4245db135da674f0b4ee0d11b9e5547594b92e7ddc53ac959f9d162c20ebbfb1.jpg)

![](images/eebde68fc7c79f63adbd4ad0d9d8d7644767d2c67d7652e797c014e206ab6835.jpg)

![](images/9fe47860da65e3928628d64787693760914b92115fe121d929cc5b1dcefce7c2.jpg)

![](images/47a71ca06c29e3faf090710d92db184e4bcfb311cb7d7e159d564e68df4506ea.jpg)

![](images/21593cf50c9766e38468ba72e73a8bfd109b0fea416709ae92ebef759a11aec0.jpg)

![](images/8d407166250c0fdae299583d0be38aa19bd83453e853689b96ddaa7bc0f36c5e.jpg)

![](images/5365ec75b388194cca033ee1fcb99638fc98ba538e6f86a70aa5d41c9922785a.jpg)

![](images/9b8e0031f890a86301885226a353db5b09af2da7e20db4666eae6ee0a38175cc.jpg)

![](images/b6fe08ce7b6c5b4f57c3109963c9c532668e5f256166f0e2b8ff98570cb91023.jpg)

![](images/a6936ffee054899af2b83bec35efba2bdbc696ecc0b57c1b791631239d7cfb8b.jpg)

![](images/05c36288f4a5b8b7f38f72bb903e8494edfdc28223ade11e66b87032f9d74406.jpg)

![](images/02c98310811a5c339b18b43ef0d340a208fcc30f2c07bf2278fc32ff87bff93f.jpg)

![](images/a0d4c5870c64264013c5910e56e919f4a7f1fe6137ede799b10b83ed69acc74f.jpg)

![](images/71b520508f6ad93dba6e21e8ab9790fc97a8164f12206d5afad4030e808ae415.jpg)

![](images/31bcf37cf2f09d58d4105fb0c842742f79f13ea8d569bc73cd1ecd9720077ddf.jpg)

![](images/6a599d5d6391466006a1aa044d7e8a634c747b034edfd48a0fc3c860d1623a19.jpg)

![](images/73eabe6e8857bde6041b87bbea7fc166dff70fe021cf85f9a4d519c05e08216e.jpg)

![](images/9db8b5b2049ec6a357603723161c95bdff647a4d1a8f032e6e5b80234092cd9a.jpg)

![](images/ce0f9d3c54a4a18d612b01bc709e84601a3fe97b35430e976e670532e732c4b3.jpg)  
Figure 3: A demo case on the BiliBili.

![](images/1e5a41bcddf085ad82374ecab00cb4375664eec1edb14b55e13c02e69cb130fa.jpg)

![](images/843dc60c1e9823aa7a587313fb67e1c1e5e743b03b31663ad5b69bbe9b619f82.jpg)

# User Task: Search and play the song "大城小事" on NetEase Cloud Music.

![](images/f1276ba79ce3ebc1fd4d11fc8225498b956b7b41c02a236fdb36d379bcf83137.jpg)

![](images/3189280b555bdcd08892bd7d32dedd66352f7120abe132db81a8a841a21fcdef.jpg)

![](images/f32ad4349ad8b3a8269529328dad9e9b6ab22f84e04cf63cf32959d33fe0b202.jpg)

![](images/a632a9efb9b04b8857e194844cc84bbee9ac9c47c22a4be4bb043a0d39daef9c.jpg)

![](images/cebfadcf5b5cb7e649c25165e95119301d97d52f430dca6771c31bd2b76f2076.jpg)

- Thought: In the search bar, search "大城小事" and click to enter the playback page.

![](images/29fb709946f0fe15f3b6cd290891c843d79be15288ea4b462cecb87ca9435083.jpg)

Thought: After entering "Big cities" in the search box, tap the search button to look for the song.

![](images/452d4eda7f3b0d06871d6e62676233fabfa57d8186c65caa0b0d815475b77f91.jpg)

- Thought: The task is to play "大城小事" via NetEase Cloud Music. The current screen shows the search results, and I need to tap the search button.

![](images/5103d31f54a4cc4294a163947d683b1567f100bf3f502243b8c13013088bd811.jpg)

Thought: The search results display multiple related songs. Select "大城小事" for playback.

![](images/27b18d246d92f1a0270245a67935ab2ec3c0d4eaf97f9ecb6628f5520d743b41.jpg)

Action: {"POINT":[356,63]}

![](images/197e08cd2a0859258952343afcd6b96b1f5eb89ba22319760bd767b047ba0381.jpg)

Action: {"TYPE":"大城小事"}

![](images/d7d13410e8bcb141b3a00cf73ac18e1ef92a2bd636f298c2cb2500fe677a2d35.jpg)

Action: {"POINT":[916,59]}

![](images/8e9f93efc017931c352e44e70839512ca80290748531d1eb80e7a273755bb81b.jpg)

Action: {"POINT":[550,370]}

![](images/c4775107ec6de64bf20375e431333384ed36955cf1944f508b9b8b03080d132a.jpg)  
Figure 4: A demo case on the NetEase Cloud Music.

![](images/66b7d04a7df9ed1b52c358720885603f06e6c8bf1239915989d3baf9f3cdc5a8.jpg)

Thought: The current screen is the playback interface of NetEase Cloud Music, showing that the song "大城小事" is playing. The task was to search for and play this song, and the interface displays playback information and control buttons, indicating that the song is being played. No further actions are required, the task is complete.

![](images/be4d8f615ab6b1568ada2f08b088355528f1d335daa65ad6bc99869a38b80c64.jpg)

Action: {"STATUS":"finish"}