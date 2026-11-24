![[Pasted image 20261024161103.png|529]]World Model（世界模型）

让机器人预测执行动作之后，环境可能发生什么变化。

比如机器人准备推动杯子，世界模型可以预测杯子可能向哪里移动、是否会碰到其他物体。

Reasoning（推理）

让机器人在执行复杂任务前进行规划、分解或中间步骤推断。

比如先判断杯子是否被其他物体挡住，再决定从哪个方向抓取。

Policy Steering（策略引导）

在生成动作的过程中，对机器人的行为进行约束或引导，使其更加符合任务目标。

比如让机器人选择碰撞风险更低的运动方式，而不是仅仅追求最快到达目标。
Pretrained Visual Repr.

预训练视觉表征

利用已有视觉模型的特征提取能力

Pretrained Lang. Model

预训练语言模型

利用已有语言理解和语义表示能力

Reinforcement Learning

强化学习

通过奖励反馈优化机器人动作策略

Dynamics Learning

动力学学习

学习机器人动作与状态变化之间的关系

三个研究方面：
1. vla的组成优化
2. 低级操作平移旋转
3. 高级策略长任务变成子任务
![[Pasted image 20261024170412.png]]
高级任务规划器可以利用高容量的模型，而低级控制策略可以专注于速度和精度，类似于分层强化学习。

eg：
		RT-1

**笔记**


具身智能强调的不只是智能体会思考，而是它的智能与身体、环境及行动后果之间存在联系。
**感知—决策—行动—反馈**的闭环。
	（拥有虚拟角色、动作能力和与环境交互的限制。所以mine craft也算是）
## 组成：
PVR = Pre-trained Visual Representation（预训练视觉表征）
图片、、视频可以用图片的方法也可以提供更丰富的3d信息
			虽然基于视频的时间CL和文本引导的预训练方法，如CLIP，提供了图像级别的信息，但它们缺乏基于MAE的自监督学习方法提供的像素级别的细节。像素级信息包含了丰富的细节信息，包括分割掩模、物体位置和深度估计等，这些信息通常对于要求高精度的机器人操作任务更有用，如VC - 1的比较[ 32 ]。DINOv2通过结合掩模图像建模、动量编码器和多作物增强来学习像素级和图像级特征，并通过OpenVLA [ 35 ]证明了在下游机器人任务中的好处。 I - JEPA关注表示空间中的块，因此比DINO等视角无关方法更有效地捕获低层图像特征。Theia将各种现成的视觉基础模型提炼成一个单一的模型，超越了孤立的个体模型，在机器人学习中对大多数现有的PVR进行了综合评估[ 38 ]。
Dynamics Learning动力学状态和动作 与pvr相似
			正向动力学学习通常比逆向动力学学习更具有挑战性，因为预测未来状态比预测过去的动作更复杂。因此，正向动力学的困难往往会带来更大的性能提升，如SMART所示。然而，逆动力学模型可以用于为只包含状态的数据集生成动作标签，例如原始的机器人操作视频。
世界模型  ：模拟出给定动作的未来状态
	LLM 
		：:LLMs蕴含丰富的世界常识知识，由此催生了众多利用该知识来改进视觉语言模型（VLAs）的方案
			eg：传统强化学习面对复杂任务时，会遇到一个问题：最终奖励非常遥远。比如制作石镐需要完成很多前置任务。如果机器人只有制作出石镐才能获得有效奖励，那么它可能探索很久也得不到有意义的反馈。一方面，AWM 将复杂目标拆成可独立学习的子目标；另一方面，LLM 引导智能体优先探索可能有用的子目标。**减少学习过程中不必要的环境交互和训练任务。** 但是存在wake去验证dream的正误
			==🟡所以世界模型可以不预测当前的状态而是类似于系统知晓一切。？==（抽象任务关系）
				世界模型不是“知道这个世界的一切”，而是拥有一个能够帮助预测、推演和规划的世界内部表示。
			==🔴目前像素级预测，状态预测，以及这个抽象任务预测==
	 Visual World Models
			生成未来状态的图像、视频或三维场景，与物理世界更加契合。它们还可进一步用于生成新的运动轨迹
						 **机器人面对被障碍物挡住的杯子：**
							**CoT 可能生成这样的高层计划：**
> 							**先移开障碍物，然后再抓取杯子。**
							**世界模型则可能帮助判断：**
> 							**如果从这个方向推动障碍物，它可能碰到杯子。**

Reasoning：
		CoT 的核心思想，是让大语言模型在给出最终答案之前，生成一系列中间推理步骤，而不是直接输出结论。
					尽管世界模型和推理方法都可以应用于低层次的控制策略和高层次的任务规划器，但目前的方法仍然是不同的。世界模型主要用于与控制政策进行交互，因为它们擅长在低级动作的情况下生成立即的下一个状态。相比之下，基于CoT的推理方法侧重于任务规划，因为它们在文本中表达了思维链，因此非常适合提炼基于文本的任务计划。
Policy sterring:
	策略指导：策略指导可以在测试时提高VLA的性能，而不需要昂贵的再培训
##  Low-Level Control Policies

通过将动作解码器与视觉编码器、语言编码器等感知模块集成，形成参数为θ的VLA模型π θ，作为执行语言指令p：的控制策略。

1 )Non-Transformer Control Policies
	非Transformer控制策略：在采用Transformer模型之前，早期针对语言条件机器人任务的控制策略在体系结构上存在显著差异。

2 )Transformer-Based Control Policies
	基于Transformer的控制策略：自从引入Transformer之后，控制策略已经收敛到类似的基于Transformer的架构。
		LLMs的最新趋势是通过生成通过API调用工具的代码来赋予其工具使用能力[[129] C. Qu, S. Dai, X. Wei, H. Cai, S. Wang, D. Yin, J. Xu, and  J. Wen, “Tool learning with large language models: a survey,” Frontiers Comput. Sci., vol. 19, no. 8, p. 198343, 2025.]。

3 )Control Policies for Multimodal Instructions
	多模态教学的控制策略：多模态教学允许新的方法来指定任务，例如通过演示，通过命名新的对象，或通过手指或鼠标点击来指定任务。

4 )Control Policies with 3D Vision
	基于3D视觉的控制策略：在我们的3D世界中，有理由假设3D视觉提供比2D图像更丰富的信息。

5 )Diffusion-Based Control Policies
	基于扩散的控制策略：基于扩散的行动生成利用了扩散模型在CV领域的成功。
	DDPM：   通过逐步去噪生成动作。
	滚动时域控制:      执行一部分动作后重新观察、重新规划。
		Visual Conditioning：   视觉条件引导  
		Time-series Diffusion Transformer：
			时间序列扩散 Transformer
		让生成的动作既符合当前环境，又在时间上连贯。
假设模型每次去噪都需要一定时间，那么总推理延迟就会受到去噪迭代次数、模型规模、输入分辨率和硬件性能等因素的影响。

这会引出几个可以研究的问题：

- 能否减少去噪步数，同时保持动作质量？
    
- 能否使用更小的去噪网络？
    
- 能否优化视觉特征提取和动作生成之间的数据传递？
    
- 能否在滚动控制过程中，兼顾动作稳定性和实时性？
6 )Diffusion-Based Control Policies with 3D Vision
	基于3D视觉的扩散控制策略：一些工作提出了将3D视觉与基于扩散的策略相结合。


7 )Control Policies for Motion Planning
	运动规划控制策略：运动规划是在满足避障、运动限制等约束的前提下，将运动任务分解为离散的路径点。

8 )Control Policies with Point-Based Actions
	基于点的行动的控制策略：最近的研究探索了利用VLMs选择或预测基于点的行动的能力- -这是构建完整VLAs的一种具有成本效益的替代方法。

9 )Large VLA
	大VLA：大VLA与RT - 2 [ 1 ]提出的原始VLA定义等价，如图2 ( a )所示。这个术语类似于LLMs和一般语言模型之间的区别，或者大型VLMs和一般VLMs之间的区别。
## 高层次任务规划器
**A .单片任务规划器。**
1 )End-to-End Task Planners
	端到端任务规划器：类似于LVLAs，任务规划器可以实现为端到端MLLMs，利用其互联网规模的知识进行任务规划。

2 )End-to-End Task Planners with 3D Vision
	具有3D视觉的端到端任务规划师：一些任务规划师也在探索3D视觉的使用。

3 )Grounded Planning
	接地任务规划器：接地任务规划涉及生成高层动作，同时考虑是否可以通过低层控制策略执行。
**B.模块化任务规划器。**
1 )Language-Based Task Planners
	基于语言的任务规划者：基于语言的方法使用自然语言描述作为交换多模态信息的媒介，如图6 ( a )所示。
2）Code-Based Task Planners
	基于代码的任务规划器：基于代码的任务规划器利用大型语言模型的编码能力，生成可执行程序形式的任务计划。对象检测器、VLM和控制策略可以通过API调用，
## DATASETS AND BENCHMARKS
A. Real-World Robot Datasets & Benchmarks
	显著的数据稀缺问题
B. Simulators, Simulated Robot Datasets & Benchmarks
	sim-to-real gap
C. Automated Dataset Collection
D. Human Datasets
E. Task Planning Benchmarks
	EgoPlan-Bench
	PlanBench
	LoTa-Bench
	Embodied agent interface
F.Embodied agent interface
	如空间推理、物理理解和世界知识