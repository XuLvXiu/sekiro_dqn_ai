
## 动机与成效

从第一期的 resnet 分类模型，到第二期的蒙特卡洛方法，到第三期时序差分 Sarsa，到第四期时序差分 Q-Learning，如今已是 DQN(Deep Q Network)，一路走来，收获很多。一方面对动作游戏有了更深层次的理解，另外一方面也学到了一些强化学习方面的知识。

在本次项目中，我们终于可以不再对 state 做更多的特征工程了，因为 DQN 的 state 空间可以是无限的，直接把游戏截图做为 state 传进去就好。

最终训练出来的 DQN policy，相比之前的 Q-Learning policy，找到的攻击机会要更少一些，忍杀也基本上没有可能了，而且稳定性也变差了一些。


## 演示视频

三周目白金的 DQN 狼 再战 稀世强者苇名弦一郎

https://www.bilibili.com/video//


## 游戏设置

- steam 开始游戏
- 游戏设置：图像设定 -- 屏幕模式设置为窗口，游戏分辨率调整为`1280*720`
- 游戏设置：按键设置 -- 使用道具设置为按键`p`, 动作（长按）吸引设置为按键 `e`. 重置视角/固定目标设置为`q`，跳跃键为`f`(未用到)，垫步/识破键为`空格`，鼠标`左键`攻击，`右键`防御
	如果需要改变按键，可以在 `config/actions_conf.yaml` 文件中依次修改各个动作中的按键.
- 把 `伤药葫芦` 设置为第一个快捷使用的道具, 最好只设置这一个道具
- 游戏设置：网络设定 -- 标题画面启动设定为`离线游玩`，目的是为了关闭残影，满地的残影烦死了。
- 保持游戏窗口在最上层，不要最小化或者被其它的全屏窗口覆盖。

## 思路与问题

### 之前项目遗留的问题
在战斗中经常会出现 boss 在防御， player 也在防御的情况，两人谁也不出招，一直僵持十几秒钟。战斗场面非常难看，也不知道 boss 在等什么，估计它的 AI 也有些问题，此时显然是应该主动攻击的。

在本次项目中，我们使用 DQN 来解决这个问题，希望能够把一部分的防御状态转为进攻，冷不丁梆梆就两拳，让战斗场面变得激烈起来。

### 游戏角色分析
弦一郎其实是个 I 人，他的攻击欲望并不强。如果 player 不主动发起攻击，boss 一般只会低频率的普通攻击，偶尔会使用前摇特别长非常容易应对的擒拿危，在这个过程中，player 几乎不会受伤，只会缓缓积累架势条；一旦 player 的攻击频率变高，boss 怒气值上来之后，它的攻击频率就会增加，而且也会使用一些比较罕见的招式来阴人， 比如下段危和原地突刺危甚至还有变种，新人遇到这种情况难免就会手忙脚乱。

### 打法: 
立足防御，识别 boss 的 3个招式(1突刺危、2擒拿危擒擒又拿拿、3飞渡浮舟)，利用破绽进行尽可能多的连击，自身架势值不高的时候鼓励进攻.

### 状态空间 state space

在之前 Q-Learning 项目中，状态有如下几种：

state-0. 默认状态，防御即可。

state-1. 突刺危，boss 先是起跳下击，接后撤突刺。处理方法为防御 + 识破 + 攻击。

state-2. 擒拿危。处理方法为攻击打断 + 攻击。

state-3. 飞渡浮舟，也就是 2 连击 + 7 连击，处理方法为防御 + 攻击。

state-4. 突刺危的变种，boss 原地后撤突刺，出招比较隐蔽。处理方法为识破 + 攻击。但是分类模型对这个分类的误召回率太高，经常把 state-0 里面的数据召回，所以只能放弃识破+攻击，改为防御。

state-5. player 受伤，有倒地和不倒地的两种情况，统一按照倒地来处理，需要按识破键快速起身，然后后撤。

state-6. boss 受伤，且 player 血量低于 60，此时后撤喝血瓶，也就是葫芦。

state-7 指的是 player 攻击对 boss 造成了伤害，此时可以发动追击，再多砍上两刀，砍死弦一郎这个老贼，砍他，浇给。但一般无法造成伤害，收益在于可以增加 boss 的架势条。

state-8 指的是 player 的架势条被打崩，此时硬直了，boss 一般会接下劈危，此时应该向后垫步吧，躲过下劈。这块处理的也不是太好。

state-10 指的是 player 架势条在下降，有可能 boss 的攻势暂时停止了，此时 player 或许可以尝试发起攻击。

state-11-19 是指攻击之后处于 state-0 的几个 step 的分段，主要目的是把它们从 state-0 中分离出来，得分会更精确。

其中，前 5 个状态由分类模型预测产生。

这一次，我们把防御状态 state-0, state-4, state-11-19 交给 DQN 来处理，但也不是全部，只有当分类模型预测出来的 class 为 0 / 4 的时候才这样做，希望它能够比 state-10 做的更好，所以它同样叫做 state-10, 原有的 state-4 state-10 state-11-19 就不再需要了，但是 state-0 还是依然存在的，比如当分类模型预测出来的  class 为 1 且信号强度不足以支持 state-1 的时候，就会 fall back 到 state-0。

剩余的几个状态全部由规则来处理，此规则来自于 Q-Learning 项目中的训练结果，这几个 state 没有 Q 值，也不参与到 DQN 的计算之中。

所以，基于规则的这几个状态对于 DQN 来说，就是终结状态，当然，player 或者 boss 死亡也同样是终结状态。

### 动作空间 action space

DQN 的可能动作只有三种：防御、普通攻击、识破+攻击，应该就可以应付全部的情况了。

但是在实际训练的时候，识破+攻击会导致 player 死的太频繁，防御时间太短(0.1s)也同样没啥意义，所以我们把动作空间简化为两种动作： 0.8秒的防御 & 普通攻击。

没错，面对一个复杂问题的时候，最好的办法就是把它简化。

总结一下，我们做的其实是如下的数据转化：

`
实时截图 image -> 分类 class(不准确) -> 状态 state(相对准确)	-> DQN	-> 动作 action(需要探索) 
																-> Rule -> 动作 action(无需探索)
`


### 问题

目前还存在一些比较明显的问题：

1. player 对空间没有感知，总是会被卡到角落里面，这是很危险的，一方面分类模型可能预测不准确，另外一方面有时候会丢失锁定，只能人工协助重新锁定 boss。

2. 代码越来越复杂，bug 也越来越多，所幸游戏本身和算法本身的容错率都很高，所以 AI 也可以玩得下去。

3. state 有点过拟合，过度的拟合了弦一郎的招式，几乎没有泛化能力，碰到其它的 boss 是打不下去的。 当然了，人类玩家打完弦一郎再去打其它 boss，同样也是泛化不了的，也需要重新拟合新 boss 的招式。

4. 训练出来的 policy 在某一个 episode 中表现不一定好，出点儿意外就挂了，但如果战斗多个 episode，那么整体结果一定是比较好的。 就类似于股票策略，它在某些交易的时候可能是亏损的，但是长期坚持下去一定是会盈利的。

5. 原地突刺危无法处理，还好它出现的概率不大，一场战斗中大概会出现 1-2 次。

6. 在战斗中仍然会出现 boss 在防御， player 也在防御的情况，两人谁也不出招，一直僵持十几秒钟。战斗场面非常难看，也不知道 boss 在等什么，估计它的 AI 也有些问题，此时显然是应该主动攻击的。

7. player 对于距离没有感知，有时候明明距离 boss 很远，也会出手攻击，自然就会打空。


## 如何训练

- 确认环境

`python debug_display_game_info.py`

会在 assets 目录中生成 debug_ui_elements.jpg，该图片中会绘制游戏屏幕截图中的各个 window 区域

同时还会弹出一个小的 tk 窗口实时显示 player 与 boss 的 hp，这个功能得感谢原作者。


- 测试：执行某个或者某几个动作

`python test.py`

可以去找不死半兵卫练习一下招式


- 训练 DQN

`python train.py`

默认会加载 checkpoint 文件中的训练相关信息以及模型参数，然后在此基础上继续训练。

进入游戏后，按 q 键锁定 boss，按 ] 键开始正式的训练，按 Backspace 键，退出程序。

如果在命令行中使用了 `--new` 参数，则会从第 0 个 episode 开始重新训练。

每一个 episode 结束之后，把训练的相关信息以及模型参数保存到 checkpoint 文件中。

在 `train.py` 中我们把 DQN 的代码拆分了出来，也就是说一部分逻辑在 `train.py` 中，另外一部分逻辑在 `dqn.py` 和 `experience_replay_memory.py` 中。

由于训练 DQN 的过程运行起来比较慢，也许换个显卡就能快起来，也许优化代码就行。 但是我们选择了另外一条路，并非在每个 step 中训练 DQN，而是把一个 episode 中的多个 step 积累起来，等到 episode 结束的时候再进行多步的训练。


- 查看 checkpoint 相关信息 

`python checkpoint.py`


## 预测

进入游戏，

在 cmd 窗口中运行：
```
python main.py 
```

等待模型加载完，tk 窗口出现，

按 q 键锁定 boss

按 ] 键, 就会针对敌方的出招自动做出预测动作了。

再次按下 ] 键，会停止预测。

在游戏过程中很有可能会卡在角落里面或因死亡而丢失视角，需要人工再重新锁定一下。

按 Backspace 键，退出程序。


## 人工备份

模型的训练过程与结果主要涉及到如下的几个文件：

- checkpoint.pth	存储了 DQN 中的两个 network的参数，以及 optimizer 的参数，以及其它一些相关信息.
- checkpoint.json	记录了当前是哪个 episode，以及完成训练时的时间

如果要训练新模型的话，可能需要对老模型的这些数据进行备份。


## 大部分代码和思路来自以下网址，感谢他们

- https://github.com/XR-stb/DQN_WUKONG
- https://github.com/analoganddigital/DQN_play_sekiro
- https://github.com/Sentdex/pygta5
- https://github.com/RongKaiWeskerMA/sekiro_play
- https://github.com/louisnino/RLcode

- https://www.lapis.cafe/posts/ai-and-deep-learning/%E4%BD%BF%E7%94%A8resnet%E8%AE%AD%E7%BB%83%E4%B8%80%E4%B8%AA%E5%9B%BE%E7%89%87%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B
- https://blog.csdn.net/qq_36795658/article/details/100533639
- https://blog.csdn.net/Guo_Python/article/details/134922730

- https://zhuanlan.zhihu.com/p/145102068
- https://zhuanlan.zhihu.com/p/630554489

- X 图片 y 按键 分类项目 https://github.com/XuLvXiu/sekiro_classifier_ai
- 蒙特卡洛项目 https://github.com/XuLvXiu/sekiro_rl_mc_ai
- 时序差分 Sarsa 项目(含 X 图片 y 人工标签分类) https://github.com/XuLvXiu/sekiro_rl_td_ai
- Q-Learning https://github.com/XuLvXiu/sekiro_rl_q_learning_ai


