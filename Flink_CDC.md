![](https://nightlies.apache.org/flink/flink-docs-master/fig/processes.svg)

# Flink 结构与原理

## JobManager

JobManager 负责整个 Flink 集群任务的调度以及资源的管理，从客户端中获取提交的应用，然后根据集群中 TaskManager 上 TaskSlot 的使用情况，为提交的应用分配相应的 TaskSlot 资源并命令 TaskManager 启动从客户端中获取的应用。

每个 Flink 至少有一个 JobManager，多 JobManager 会成为 master - slave 结构，会选取一个 JobManager 作为 leader。

### Resource Manager

负责管理 Flink 集群中的计算资源，其中主要来自于 TaskManager。ResourceManager 会动态接收 SlotRequest。如果采用 Native 模式部署，会动态的向集群资源管理器申请 Container 并启动 TaskManager。

### Dispatcher

提供 REST 接口用于 Flink 应用执行并且启动一个新的 JobMaster 为每一个提交的 job。并提供一个 WebUI 用于提供 job 的执行信息。

### JobMaster

单个 JobMaster 负责管理单个 JobGraph 的执行，多个 job 可以同时在一个 Flink cluster 内部执行，每个 job 拥有自己的 JobMaster

## TaskManager

Flink 负责集群执行 job 服务的工作，每个 TaskManager 内部包含一个或多个 Slot，负责承载 Task

## 客户端 JobGraph 的创建

客户端调用 StreamExecutionEnvironment.getExecutionEnvironment() 方法，并且添加好各种 DataStream 操作之后，调用 StreamExecutionEnvironment.execute() 方法产生 StreamGraph，再转化为 JobGraph 数据结构。

+ 先借助 DataStream API 构建 Transformation 操作集合，通过一系列遍历、构建操作获取 StreamNode 和 StreamEdge 构成的 StreamGraph
+ 遍历 StreamGraph，对 StreamNode 的上下游结构进行嵌合判断，并生成 JobVertex 以及制定 slotSharingGroup 等，并配置 Checkpoint 等配置

### StreamNode 嵌一起的条件

1. 下游节点只有一个输入
2. 下游节点操作符不为 null
3. 上游节点操作符不为 null
4. 上下游节点在一个 slotSharingGroup 内
5. 下游节点的连接策略是 ALWAYS
6. 上游节点的连接策略是 HEAD 或 ALWAYS
7. edge 的分区函数是 FowardPartitioner 的实例
8. 上下游节点并行度相等
9. 可以进行节点连接操作

## 客户端提交 JobGraph

用户打包生成 jar 包，`flink run` 命令初始化客户端 main 程序，客户端将提交的应用程序打包成 PackagedProgram，其中包含 mainClass、classpaths 等信息，并根据不用服务配置加载创建 ContextEnvironment，通过反射的方式执行 jar 应用程序中的 main() 方法。最后用过 ContextEnvironment 创建 ExecutionEnvironment 或者 StreamExecutionContextEnvironment 最后构建 JobGraph 并提交集群。

## 集群解析与 Task 执行

集群收到 JobGraph 再到执行任务中间还需要经过两个步骤，分别是从 JobGraph 转为 ExecutionGraph 以及 ExecutionGraph 的调度与执行

### JobGraph - ExecutionGraph

![](https://nightlies.apache.org/flink/flink-docs-master/fig/job_and_execution_graph.svg)

客户端将 JobGraph 提交到集群后，集群通过 Dispatcher 组建接收。Dispatcher 组建通过 JobManagerRunnerFactory 创建 JobManagerRunner 实例，通过 JobManagerRunner 启动服务。JobManager 服务底层主要通过 JobMaster 实现，负责整个生命周期和 Task 的调度工作。

### ExecutionGraph

JobMaster 启动完毕从 ResourceManager 中申请到 Slot 计算资源后，调度和执行 Job 中的 SubTask 任务。任务会先经过 JobMaster 的一系列检查和封装后，发送给下游的 TaskManager，TaskManager 通过 TaskDeploymentDescriptor 的信息启动 Task 实例，TaskExecutor.submitTask() 方法根据提交的信息创建并运行 Task 线程。并且 TaskExecutor 中 Task 会实现 Runnable 接口，可以被线程池调度和执行。通过 Task 中 每个 Job 的 currentState 对 Job 的状态进行监控，并且 Job 也会根据这个状态判断下一步、状态转换以及是否能重启等功能的判断。

### 线程模型

[Actor model](https://en.wikipedia.org/wiki/Actor_model)

[Akka](https://www.baeldung.com/akka-actors-java)

[Actor 模型](https://dl.acm.org/doi/10.5555/1624775.1624804) 是 1973 年提出的一种并发编程的思想，其原理在于：actors 可以修改自身的私有状态，但是仅能通过传递信息的方式间接的影响其他 actors 的状态。这样的做法可以摒弃基于锁的同步的需求(removing the need for lock-based synchronization)。其动机是由数十、数百甚至数千个独立微处理器组成的高度并行计算机的前景，每个微处理器都有自己的本地存储器和通信处理器，通过高性能通信网络进行通信。

与面向对象的原理类似，Actor 模型将每个单元都看作一个 actor，每个 actor 都是一个计算单元，对于每个接收到的信息，可以并发的实现以下操作：

+ 可以发送有限的数量的信息给其他 actor
+ 可以创建有限数量的新 actor
+ 指定要用于它接收的下一条消息的行为

#### Flink 与 Akka

[Akka and Actors](https://cwiki.apache.org/confluence/display/FLINK/Akka+and+Actors)

Flink 分布式通讯是通过 Akka 实现的。有了 Akka，所有远程过程调用现在都实现为异步消息。这主要影响组件 JobManager、TaskManager 和 JobClient。

Flink系统由三个相互通信的分布式组件组成：JobClient、JobManager 和 TaskManager。JobClient 从用户那里获取一个 Flink 作业，并将其提交给 JobManager。然后 JobManager 负责安排作业的执行。首先，它分配所需的资源量。这主要包括 TaskManagers 上的执行槽。

![](https://cwiki.apache.org/confluence/download/attachments/53741538/jobExecutionProcess.png?version=1&modificationDate=1426878738000&api=v2)

> The string akka://flink/user/taskmanager is an actor path, which indicates where an actor resides in the actor supervision hierarchy: flink is the name of the actor system, and user is the guardian actor for all user-created top-level actors. The TaskManager actor is directly beneath the guardian actor, which means that TaskManager is a top-level actor that is created via the `ActorSystem.actorOf` method.
>

#### Failure Detection

Flink 通过 Akka 的 DeathWatch 机制检测故障组件。DeatchWatch 允许 Actor 观察其他 Actor，即使他们不受该 Actor 监管，甚至活跃在不同的 Actor 系统中（这部分的设计与 Flink Task 重启策略相关）。在 Flink 中，JobManager 监视所有注册的 TaskManager，TaskManager 监视 JobManager。这样，两个组件都知道另一个组件何时不再可访问。JobManager 的反应是将相应的TaskManager 标记为 dead，以防止将来的任务部署到它。此外，它会使当前在该任务管理器上运行的所有任务失败，并重新安排在其他 TaskManager 上执行。如果 TaskManager 只是因为临时连接丢失而被标记为无效，那么一旦重新建立连接，它就可以简单地在 JobManager 中重新注册。TaskManager 还监视 JobManager。这种监视允许 TaskManager 在检测到失败的 JobManager 时，通过使所有当前运行的任务失败，进入干净状态。此外，如果触发的死亡仅由网络拥塞或连接丢失引起，TaskManager 将尝试重新连接到 JobManager。

### Task 重启与容错

重启策略分三种类型：

+ 固定延时重启
+ 按失败率重启
+ 无重启

# Flink RPC 通信框架

Flink RPC 通信框架是基于 Akka 系统

## Actors in Flink

一个 Actor 自身是一个带有状态和行为的容器。它的 actor 线程会顺序的处理接收到的消息。它减轻了用户锁定和线程管理这一容易出错的任务，因为一个参与者一次只有一个线程处于活动状态。然而一个必须保证的前提是，一个 Actor 的内部状态只能由当前 Actor 线程访问到。Actor 的行为是由接收函数定义的，该函数为每个消息包含在接收该消息时执行的一些逻辑。

Actor 由状态、行为和邮箱三部分组成

+ 状态：Actor 对象的变量信息，由 Actor 自己管理，避免了并发环境下的锁和内存原子性等问题
+ 行为：指定 Actor 中的计算逻辑，通过接收到的消息改变 Actor 的状态
+ 邮箱：每个 Actor 都有自己的邮箱，通过邮箱能简化锁及线程管理。邮箱是 Actor 之间的通信桥梁，邮箱内部通过 FIFO 消息队列存储 Actor发送方的消息，Actor接收方从邮箱队列中获取消息

Flink系统由三个相互通信的分布式组件组成：JobClient、JobManager 和 TaskManager。JobClient 从用户那里获取一个 Flink 作业，并将其提交给 JobManager。然后 JobManager 负责安排作业的执行。首先，它分配所需的资源量。这主要包括 TaskManagers 上的执行槽。

资源分配后，JobManager 将作业的各个任务部署到相应的 TaskManager。收到任务后，TaskManager 会生成一个执行任务的线程。状态更改（如开始计算或完成计算）会发送回 JobManager。根据这些状态更新，JobManager 将引导作业执行，直到作业完成。作业完成后，它的结果将被发送回 JobClient，后者将告诉用户。

#### Asynchronous vs. Synchronous Messages

在任何可能的情况下，Flink 都会尝试使用异步消息，并将响应作为后续处理。后续和少数现有的阻塞调用会有一个超时，超过该超时后操作将被视为失败。这样可以防止系统在消息丢失或分布式组件崩溃时出现死锁。然而，如果碰巧有一个非常大的集群或慢速网络，则可能会错误地触发超时。因此，这些操作的超时可以通过配置中的“akka.aask.timeout”指定。

在一个参与者可以与另一个参与者交谈之前，它必须为其检索(retrieve)一个 ActorRef。此操作的查找也需要超时。为了在参与者未启动时使系统快速故障，查找超时设置为比常规超时更小的值。如果遇到查找超时，可以通过配置中的“akka.ulookup.timeout”来增加查找时间。

Akka 的另一个特点是它设置了可以发送的最大消息大小的限制。原因是它保留了相同大小的序列化缓冲区，并且不想浪费内存。如果因消息超过最大大小而遇到传输错误，可以通过配置中的"akka.framesize"来增加帧大小。

### 创建 Akka 系统

Akka 系统核心组件包括 ActorSystem 和 Actor，构建 Akka 系统需要创建 ActorSystem，然后通过 ActorSystem 创建 Actor。

### Flink RPC 与 Akka 的关系

集群运行时中实现了 RPC 通信节点功能的主要有 Dispatcher、ResourceManager、TaskManager 和 JobMaster 等组件。集群的 RPC 服务组件是 RpcEndpoint，每个 RpcEndpoint 包含一个内置的 RpcServer 负责执行本地和远程的代码请求，RpcServer 对应 Akka 中的 Actor 实例。

RpcEndpoint 中创建和启动 RpcServer 主要是基于集群中的 RpcService 实现，RpcService 的主要实现是 AkkaRpcService。AkkaRpcService 将 Akka 中的 ActorSystem 进行封装，通过 AkkaRpcService 可以创建 RpcEndpoint 中的 RpcServer，同时基于 AkkaRpcService 提供的 connect() 方法与远程 RpcServer 建立 RPC 连接提供远程进程调用的能力。

### 运行时 RPC

除了基于 Akka 构建底层通信系统外，还是用 JDK 动态代理构建 RpcGateway 接口的代理类

### RpcEndpoint 的设计与实现

每个 RpcEndpoint 都对应一个由 endpointId 和 actorSystem 确定的路径，该路径对应同一个 Akka Actor，所有需要实现 RPC 通信的集群组件都会继承 RpcEndpoint 抽象类，例如 TaskExecutor, Dispatcher 以及 ResourceManager 组件服务。

结构如下：

+ RpcService：RpcEndpoint 的后台管理服务
+ RpcServer：RpcEndpoint 的内部服务类
+ MainThreadExecutor：封装了 MainThreadExecutable 接口，主要底层实现是 AkkaInvocationHandler 代理类。所有 本地和远程 RpcGateway 执行请求都会通过动态代理类的形式转换到 AkkaInvocationHandler 代理类中执行

## AkkaRpcService

### AkkaRpcService 创建和初始化

RpcService 负责创建和启动 Flink 集群环境中 RpcEndpoint 组件的 RpcServer。AkkaRpcService 作为 RpcService 的唯一实现类，基于 Akka 的 ActorSystem 进行封装，为不同的 RpcEndpoint 创建相应的 ActorRef 实例

### AkkaRpcService 初始化 RpcServer

创建 RpcEndpoint 组件的时候，会在 RpcEndpoint() 构造方法中调用 AkkaRpcService.startServer(this) 方法初始化 RpcEndpoint 对应的 RpcServer。启动过程主要包含两个部分：创建 Akka Actor 引用类 ActorRef 实例和 InvocationHandler 动态代理

RpcServce 创建过程中，包含了创建 RpcEndpoint 中的 Actor 引用类 ActorRef 和 AkkaInvocationhandler 动态代理类，最后将动态代理类转换为 RpcServer 接口返回给 RpcEndpoint 实现类，此时实现的组件就能够获取到 RpcServer 服务，且通过 RpcServer 代理了所有的 RpcGateways 接口。

### 通过 AkkaRpcService 连接 RpcServer 并创建 RpcGateway

当 AkkaRpcService 启动 RpcEndpoint 中的 RpcServer 后，RpcEndpoint 组件仅能对外提供处理 RPC 请求的能力，RpcEndpoint 组件需要在启动后向其他组件注册自己的 RpcEndpoint 信息，并完成组件之间的 RpcConnection 注册，才能相互访问和通信

## RpcServer 动态代理

RpcServer 中提供 RpcGateway 接口方法，最终都会通过 AkkaInvocationHandler.invoke() 方法进行代理实现。AkkaInvocationHandler 中根据在本地执行还是远程执行将代理方法进行区分。通常情况下，RpcEndpoint 实现类除了调用指定服务组件的 RpcGateway 接口之外，其余的 RpcGateway 接口基本上都是本地调用和执行的。

本地接口主要有 AkkaBasedEndpoint, RpcGateway，StartStoppable，MainThreadExecutable 和 RpcServer 等

远程调用会在 AkkaInvocationHandler 中创建 RpcInvocationMessage，并通过 Akka 发送 RpcInvocationMessage 到指定地址的远端进程中，远端的 RpcEndopoint 会接收 RpcInvocationMessage 并进行反序列化，然后调用底层的动态代理类实现进程内的方法调用

## AkkaRpcActor 的设计与实现

在 RpcEndpoint 中创建的 RemoteRpcInvocation 消息，最终会通过 Akka 系统传递到被调用房，例如 TaskExecutor 向 ResourceManager 发送 SlotReport 请求时会在 TaskExecutor 中将 ResourceManagerGateway 的方法名称和参数打包成 RemoveRpcInvocation 对象，然后经过网络发送到 ResourceManager 中的 AkkaRpcActor，在 ResourceManager 本地执行具体方法。

首先在 AkkaRpcActor 中创建 Receive 对象，用于处理 Akka 系统接收的其他 Actor 发送过来的信息。通过 Akka 的 ReceiveBuilder 工厂类创建了 RemoteHandshakeMessage、ControlMessages 等消息对应的处理器，其中 RemoteHandshakeMessage 主要用于进行正式 RPC 通信之前的网络连接检测。ControlMessages 用于控制 Akka 系统，例如启动和停止 Akka Actor 等控制消息。

在 AkkaRpcActor.handleMessage() 方法中，最终会调用 handleRpcMessage() 方法继续对 RPC 消息进行处理，此时会根据传入 RPC 消息进行判别，确定消息是否为 RunAsync、CallAsync 以及 RpcInvocation 等对象类型。如果是 RunAsync 或 CallAsync 等线程实现，则直接调用 handleRunAsync() 或 handleCallAsync() 方法将代码块提交到本地线程池中执行。对于 RpcInvocation 类型消息，则会调用 handleRpcInvocation() 方法。

## 集群组件之间的 RPC 通信

TaskExecutor 启动后会立即向 ResourceManager 中注册当前 TaskManager 的信息。同样 JobMaster 启动后也立即会向 ResourceManager 注册 JobMaster 的信息。这里的注册连接在 Flink 中被称为 RegisteredRpcConnection，集群组件之间的 RPC 通信都会通过创建 RegisteredRpcConnection 进行，例如获取 RpcEndpoint 对应的 RpcGateway 接口以及维护组件之间的心跳连接等。

集群运行时中各组件的注册连接主要通过 RegisteredRpcConnection 基本类提供，且实现子类主要有 JobMangerRegisteredRpcConnection、ResourceManagerConnection 和 TaskExecutorToResourceManagerConnection 三种

---

集群运行时，各组件之间都是基于 RPC 通信框架相互访问。RpcEndpoint 组件会创建于其他 RpcEndpoint 之间的 RegisteredRpcConnection，并通过 RpcGateway 接口的动态代理于其他组件进行通信，底层通信则以来 Akka 通信框架实现。并且通过大量的代理类，实现了通信框架的实现类，以便未来可以灵活的底层 RPC 框架的替换。

# 状态管理

## KeyedState 与 OperatorState

| | keyedState | OperatorState |
| --- | :---: | :---: |
| 使用算子类型 | 用于 KeyedStream 中的算子 | 可用于所有类型的算子 |
| 状态分配 | 每个 Key 对应一个状态，单个 Operator 中可能具有多个 KeyedState | 单个 Operator 对应一个 算子状态 |
| 创建和访问方式 | 重新 RichFunction，访问 RuntimeContext 对象 | 实现 CheckpointedFunction 或 ListCheckpointed 接口 |
| 横向拓展 | 状态随着 Key 再多个算子实例上迁移 | 多种状态重新分配方式 |
| 支持数据类型 | ValueState, ListStats, ReducingState, AggregatingState, MapState | ListState, BroadcastState |

### 初始化流程

TaskManager 启动 Task 后，会调用 StreamTask.invoke() 出发当前 Task 中算子的执行，在 invoke() 方法中会先调用 beforeInvoke() 方法，初始化 Task 中所有 Operator，其中包括创建和初始化算子中的状态数据。

StreamTask 调用 initializeStateAndOpen() 方法对当前 Task 所有算子的状态数据进行初始化，首先会获取 StreamTask.OperatorChain 中所有 Operator，然后对每个 Operator 进行状态初始化，初始化完毕才会调用 StreamOperator.open() 并接入数据进行处理

### HeapKeyedStateBackend

1. KeyedStateBackend 分别继承了 PriorityQueueSetFactory 和 KeyedStateFactory 接口
2. KeyedStateBackend 具有 AbstractKeyedStateBackend 基本类实现，在 AbstractKeyedStateBackend 中实现了 CheckpointListener 接口，用于向 CheckpointCoordinator 汇报当前 StateBackend 所有算子 Checkpoint 完成情况
3. AbstractKeyedStateBackend 实现了 SnapshotStrategy 接口，提供了 snapshot() 方法，对 KeyedStateBackend 中的状态数据进行快照，将状态数据写入外部文件
4. AbstractKeyedStateBackend 的实现类主要有 HeapKeyedStateBackend 和 RocksDBKeyedStateBackend。HeapKeyedStateBackend 借助 JVM 堆内存存储 KeyedState 状态数据，RocksDBKeyedStateBackend 则借助 RocksDB 管理的堆外内存存储 KeyedState 状态数据

## OperatorState

OperatorStateBackend 实现了 OperatorStateStore、SnapShotStrategy、Closable 和 Disposable 四个接口

+ OperatorStateStore 接口提供了获取 BroadcastState、ListState 以及注册在 OperatorStateStore 中的 StateNames 的方法
+ SnapshotStrategy 接口提供了对状态数据进行快照操作的方法，用于 Checkpoint 操作中对 OperatorStateBackend 中的状态数据进行快照
+ Closeable 和 Disposable 接口则分别提供了关闭和销毁 OperatorStateBackend 的操作

OperatorStateBackend 具有默认基本实现类 DefaultOperatorStateBackend，这和 KeyedStateBackend 有所不同，KeyedStateBackend 的实现类有 HeadKeyedStateBackend 和 RocksDBKeyedStateBackend 两种类型。从 DefaultOperatorStateBackend 中可以看出，所有算子的状态数据都只能存储在 JVM 堆内存中

## StateBackend

StateBackend 作为状态存储后端，提供了创建和获取 KeyedStateBackend 及 OperatorStateBackend 的方法，并通过 CheckpointStorage 实现了对状态数据的持久化存储。Flink 支持 MemoryStateBackend、FsStateBackend 和 RocksDBStateBackend 三种类型的状态存储后端，三者的主要区别在于创建的 KeyedStateBackend 及 CheckpointStorage 不同。例如，MemoryStateBackend 和 FileStateBackend 创建的是 HeapKeyedStateBackend，RocksDBStateBackend 创建的是 RocksDBKeyedStateBackend。本节我们重点来看 StateBackend 的设计与实现。

## Checkpoint

[checkpoints](https://nightlies.apache.org/flink/flink-docs-release-1.17/docs/ops/state/checkpoints/)

[fault-tolerance checkpointing](https://nightlies.apache.org/flink/flink-docs-release-1.17/docs/dev/datastream/fault-tolerance/checkpointing/)

Flink中主要借助Checkpoint的方式保障整个系统状态数据的一致性，也就是基于ABS算法实现轻量级快照服务

### 原理

Checkpoint 的执行过程分为三个阶段：启动、执行以及确认完成。其中 Checkpoint 的启动过程由 JobManager 管理节点中的 CheckpointCoordinator 组件控制，该组件会周期性地向数据源节点发送执行 Checkpoint 的请求，执行频率取决于用户配置的 CheckpointInterval 参数

首先在 JobManager 管理节点通过 CheckpointCoordinator 组件向每个数据源节点发送 Checkpoint 执行请求，此时数据源节点中的算子会将消费数据对应的 Position 发送到 JobManager 管理节点中。然后 JobManager 节点会存储 Checkpoint 元数据，用于记录每次执行 Checkpoint 操作过程中算子的元数据信息，例如在 FlinkKafkaConsumer 中会记录消费 Kafka 主题的偏移量，用于确认从 Kafka 主题中读取数据的位置。最后在数据源节点执行完 Checkpoint 操作后，继续向下游节点发送 CheckpointBarrier 事件，下游算子通过对齐 Barrier 事件，触发该算子的 Checkpoint 操作。当下游的 map 算子接收到数据源节点的 Checkpoint Barrier 事件后，首先对 Block 当前算子的数据进行处理，并等待其他上游数据源节点的 Barrier 事件到达。该过程就是 Checkpoint Barrier 对齐，目的是确保属于同一 Checkpoint 的数据能够全部到达当前节点。Barrier 事件的作用就是切分不同 Checkpoint 批次的数据。当 map 算子接收到所有上游的 Barrier 事件后，就会触发当前算子的 Checkpoint 操作，并将状态数据快照到指定的外部持久化介质中，该操作主要借助状态后端存储实现。接下来，状态数据执行完毕后，继续将 Barrier 事件发送至下游的算子，进行后续算子的 Checkpoint 操作。另外，在 map 算子中执行完 Checkpoint 操作后，也会向 JobManager 管理节点发送 Ack 消息，确认当前算子的 Checkpoint 操作正常执行。此时 Checkpoint 数据会存储该算子对应的状态数据，如果 StateBackend 为 MemoryStateBackend，则主要会将状态数据存储在 JobManager 的堆内存中。像 map 算子节点一样，当 Barrier 事件到达 sink 类型的节点后，sink 节点也会进行 Barrier 对齐操作，确认上游节点的数据全部接入。然后对接入的数据进行处理，将结果输出到外部系统中。完成以上步骤后，sink 节点会向 JobManager 管理节点发送 Ack 确认消息，确认当前 Checkpoint 中的状态数据都正常进行了持久化操作。 当所有的 sink 节点发送了正常结束的 Ack 消息后，在 JobManager 管理节点中确认本次 Checkpoint 操作完成，向所有的 Task 实例发送本次 Checkpoint 完成的消息。

### 存储

Flink Checkpoint 存储分为两种方式

+ _<font style="color:rgb(0, 0, 0);">JobManagerCheckpointStorage</font>_
+ _<font style="color:rgb(0, 0, 0);">FileSystemCheckpointStorage</font>_

<font style="color:rgb(0, 0, 0);">当 FileSystemCheckpointStorage 被启用时，数据将会被存储进文件系统，例如 s3、hdfs、普通文件存储等，否则，数据将会通过 JobManagerCheckpointStorage 存储进 JobManager 的</font>堆内存<font style="color:rgb(0, 0, 0);">中</font>

## State Snapshots

[state snapshots](https://nightlies.apache.org/flink/flink-docs-release-1.17/docs/learn-flink/fault_tolerance/#state-snapshots)

### 定义

+ Snapshot：一个通用术语，指的是 Flink 作业状态的全局一致图像。快照包括指向每个数据源的指针（例如，指向文件或 Kafka 分区的偏移量），以及来自作业的每个有状态运算符的状态副本，该状态副本是由于处理了直到源中这些位置的所有事件而产生的。
+ Checkpoint：Flink 自动生成的快照，用于从故障中恢复。检查点可以是增量的，并且经过优化可以快速恢复。
+ Externalized Checkpoint；通常情况下，检查点不会被用户操纵。Flink 在作业运行时只保留最近的 n 个检查点（n 是可配置的），在作业取消时删除它们。但是您可以将它们配置为保留，在这种情况下，您可以从它们手动恢复。
+ Savepoint：用户（或 API 调用）出于某些操作目的手动触发的快照，例如有状态的重新部署/升级/缩放操作。保存点始终是完整的，并针对操作灵活性进行了优化。

### 原理

#### Chandy-Lamport algorithm

[Chandy-Lamport algorithm](https://en.wikipedia.org/wiki/Chandy%E2%80%93Lamport_algorithm)

#### flink async barrier snapshotting

![](https://nightlies.apache.org/flink/flink-docs-release-1.17/fig/stream_barriers.svg)

当 CheckpointCoordinator（ JobManager 的一部分）指示 JobManager 开始检查点时，它会让所有源记录它们的偏移量，并将编号的检查点屏障插入到它们的流中。这些屏障流经作业图，指示每个检查点之前和之后的流部分。

检查点 n 将包含每个操作符的状态，这些操作符是由于消耗了检查点屏障 n 之前的每个事件而导致的，而没有消耗之后的任何事件。

当作业图中的每个操作员接收到其中一个障碍时，它会记录其状态。具有两个输入流的操作员（如 CoProcessFunction）执行屏障对齐，以便快照将反映从两个输入流向（但不超过）两个屏障的消耗事件所产生的状态。

![](https://nightlies.apache.org/flink/flink-docs-release-1.17/fig/stream_aligning.svg)

### 反压下的 Checkpoint

[checkpointing under backpressure](https://nightlies.apache.org/flink/flink-docs-release-1.17/docs/ops/state/checkpointing_under_backpressure/)

1. 通过优化 Flink 作业、调整 Flink 或 JVM 配置或扩大规模来消除背压源。
2. 减少 Flink 作业中缓冲的飞行中数据量。
3. 启用未对齐的 Checkpoint。

#### Buffer Debloating

[Network Buffer Debloating](#1cb88414)

缓冲区去浮动机制可以通过将属性 taskmanager.network.memory.buffer-debloat.enabled 设置为 true 来启用。

此功能适用于对齐和未对齐的 Checkpoint，并且在这两种情况下都可以提高 Checkpoint 设置时间，但通过对齐的 Checkpoint 将最明显地看到去浮动的效果。当使用未对齐 Checkpoint 的缓冲区去浮动时，额外的好处是 Checkpoint 大小更小，恢复时间更快（需要持久化和恢复的运行中数据更少）。

#### Unaligned Checkpoints

从Flink 1.11开始，Checkpoint 可以不对齐。未对齐的 Checkpoint 包含运行中的数据（即存储在缓冲区中的数据），作为 Checkpoint 状态的一部分，允许 Checkpoint 屏障超越这些缓冲区。因此， Checkpoint持续时间变得与当前吞吐量无关，因为 Checkpoint 屏障不再有效地嵌入到数据流中。

如果由于 backpressure 导致 Checkpoint 持续时间非常长，则应该使用未对齐的 Checkpoint。然后，Checkpoint 时间基本上与端到端延迟无关。请注意，未对齐的 Checkpoint 会增加状态存储的 I/O，因此当状态存储的 IO 实际上是 Checkpoint 期间的瓶颈时，不应该使用它。

#### Aligned Checkpoint Timeout

激活后，每个 Checkpoint 仍将作为已对齐的 Checkpoint 开始，但当全局 Checkpoint 持续时间超过已对齐 Checkpoint 超时时，如果已对齐 Checkpoint 将未完成，则 Checkpoint 将作为未对齐 Checkpoint 继续。

# 网络通信

## Network Stack

[Data Exchange Between Tasks](https://cwiki.apache.org/confluence/display/FLINK/Data+exchange+between+tasks)

Flink 集群中 TaskManager 之间的数据交换，以及运行在不同 TaskManager 之间的 Task 实例也会发生数据交换，而这些数据交换主要靠 NetworkStack 实现。除了提供高效的网络I/O，还提供了灵活的反压机制。

![](https://cwiki.apache.org/confluence/download/attachments/53741520/jobmanager-taskmanagers.png?version=1&modificationDate=1426848220000&api=v2)

NetworkStack 在不同 TaskManager 之间建立 TCP 连接，主要依赖 Netty 通信框架实现。

TaskManager 中会运行多个 Task 实例，例如在 TaskManager1 中运行了 TaskA-1 和 TaskA-2， 在 TaskManasger2 中运行了 TaskB-1 和 TaskB-2，TaskA 中从外部接入数据并处理后，会通过基于 Netty 构建的 TCP 连接发送到 TaskB 中继续处理。从上游的 TaskA 实例来讲，经过 Operator 处理后的数据，通过 RecordWriter 组件写入网络栈，即算子输出的数据并不是直接写入网络，而是先将数据元素转换为二进制 Buffer 数据，并将 Buffer 缓存在 ResultSubPartition 队列中，再从 ResultSubPartition 队列将 Buffer 数据消费后写入下游 Task 对应的 InputChannel。在上游的 Task 中会创建 LocalBufferPool 为数据元素申请对应的 Buffer 的存储空间，且上游的 Task 会创建 NettyServer 作为网络连接服务端，并与下游 Task 内部的 NettyClient 之间建立网络连接。

对下游的 Task 实例来讲，会通过 InputGate 组件接收上游 Task 发送的数据，在 InputGate 包含多个 InputChannel。InputChannel 实际上是将 Netty 中 Channel 进行封装，数量取决于 Task 的并行度。上游 Task 的 ResultPartition 会根据 ChannelSelector 选择需要将数据下发到哪一个 InputChannel 中，其实现类似 Shuffe 的数据洗牌操作。

![](https://cwiki.apache.org/confluence/download/attachments/53741520/transfer.png?version=1&modificationDate=1426849435000&api=v2)

这张图展示了数据从 MapDriver 中产生后经过一系列的处理最后被发送给 ReduceDriver 的过程。数据产生后，被发送到 RecordWriter 对象，由 RecordWriter 内包含的多个 RecordSerializer 对象进行序列化操作，并将数据放入固定大小的缓冲区，每个会消费这些数据的下游 Task 会对应一个单独的 RecordSerializer。ChannelSelector 选择一个或多个序列化程序来放置记录。例如，如果记录是广播的，它们将被放置在每个序列化程序中。如果记录是哈希分区的，ChannelSelector 将评估记录上的哈希值，并选择适当的序列化程序。

序列化程序将记录序列化为它们的二进制表示，并将它们放在固定大小的缓冲区中（记录可以跨越多个缓冲区）。这些缓冲区被移交给 BufferWriter，并被写入ResultPartition（RP）。RP由几个子分区（ResultSubpartitions —— RS）组成，这些子分区为特定的使用者收集缓冲区。在图片中，缓冲区的目的地是第二个 reducer（在 TaskManager2 中），它被放置在 RS2 中。由于这是第一个缓冲区，RS2 可供使用（请注意，此行为实现了流式混洗），并通知 JobManager 这一事实。

JobManager 查找 RS2 的使用者，并通知 TaskManager2 有大量数据可用。到 TaskManager2 的消息向下传播到应该接收这个缓冲区的 InputChannel，后者反过来通知 RS2 可以启动网络传输。然后，RS2 将缓冲区移交给TM1的网络堆栈，TM1 再将其移交给 Netty 进行运输。网络连接是长时间运行的，存在于 TaskManager 之间，而不是单个任务之间。

一旦 TaskManager2 接收到缓冲区，它就会通过类似的对象层次结构，从 InputChannel（相当于IRPQ的接收器端）开始，到 InputGate（包含几个IC），最后到达 RecordDeserializer，该 RecordDeserilizer 从缓冲区生成类型化的记录，并将它们移交给接收任务。

## Network Memory Tuning

[Network Memory Tunning Guide](https://nightlies.apache.org/flink/flink-docs-release-1.17/docs/deployment/memory/network_mem_tuning/)

### Buffer Debloating 机制

以前，配置飞行中数据量的唯一方法是指定 buffer 量和 buffer 大小。然而，理想值可能很难选择，因为每个部署的理想值都不同。Flink 1.14中添加的 buffer debloating 机制试图通过自动将飞行中的数据量调整到合理的值来解决这个问题。

buffer 去浮动功能计算子任务的最大可能吞吐量（在总是繁忙的情况下），并调整飞行中数据的量，使这些飞行中数据消耗的时间等于配置的值。

可以通过将属性 taskmanager.network.memory.buffer-debloat.enabled 设置为 true 来启用 buffer debloating 冲机制。消耗飞行中数据的目标时间可以通过将 taskmanager.network.memory.buffer-debloat.target 设置为持续时间来配置。默认值就可以应对大多数的场景。

该功能使用过去的吞吐量数据来预测消耗剩余飞行中数据所需的时间。如果预测不正确，去浮动机制可能会以以下两种方式之一失败：

+ 将没有足够的缓冲数据来提供全吞吐量。
+ 将有太多缓冲的飞行中数据，这将对对齐的检查点屏障传播时间或未对齐的检查点通过大产生负面影响。

如果作业中有不同的负载（即传入记录的突然峰值、定期触发窗口聚合或联接），则可能需要调整以下设置：

+ taskmanager.network.memory.buffer-debloatperiod：这是重新计算缓冲区大小之间的最短时间段。周期越短，去浮机制的反应时间就越快，但用于必要计算的CPU开销就越高。
+ taskmanager.network.memory.buffer-debloat.samples：这会调整吞吐量测量平均值的样本数。可以通过taskmanager.network.memory-buffer-defloatperiod调整收集的样本的频率。样本越少，去浮动机制的反应时间就越快，但吞吐量突然飙升或下降的可能性就越高，这可能会导致缓冲区去浮动机制误判飞行中数据的最佳量。
+ taskmanager.network.memory.buffer-debloat.threshold-percentages：用于防止频繁更改缓冲区大小的优化（即，如果新大小与旧大小相比没有太大差异）。

有关更多详细信息和附加参数，请参阅配置文档。

以下是可用于监控当前缓冲区大小的指标：

+ estimatedTimeToConsumeBuffersMs：消耗所有输入通道数据的总时间
+ debloatedBufferSize：当前缓冲区大小

总结起来，可以认为是 Flink 在 buffer 池的大小不再随时改变，而是会根据固定时间重新计算 buffer 池的浮动获取部分的大小来适应网络开销。

## 反压机制

当下游 Task 处理速度下降或者因为某些意外因素阻塞时，上游 Task 继续发送 Buffer 数据就会产生下游 Task 数据堆积无法及时处理的问题，极端情况下可能导致整个系统宕机。为了避免这种情况的发生，就需要借助反压的机制来避免这个问题的发生。

### 早期基于 TCP 的反压机制

早期的 Flink 并没有单独实现反压机制，而是依靠的 TCP 中的反压机制，如果 TCP Channel 中的数据没有被及时处理，就会影响上游节点数据的写出操作。TaskManager 之间会启动一个 TCP 通道，所有 Task 都会通过多路复用使用用一个 TCP 通道，上游 Netty 通过 channelWritablilityChanged() 方法感知当前通道能否写入。但这有个问题：

1. 下游 Task 处理能力不足产生了反压会导致整个 TCP 通道的堵塞，导致整个 TaskManager 所有 Task 都无法传输 Buffer 数据
2. 上游只能用过 TCP 通道状态被动感知下游处理能力，不能提前调整发送频率，不能根据当前数据积压情况及时调整数据处理速度

[how flink handles backpressure](https://www.ververica.com/blog/how-flink-handles-backpressure)

[buffer pools](https://www.ververica.com/hs-fs/hubfs/Imported_Blog_Media/buffer-pools-1.jpg?width=600&name=buffer-pools-1.jpg)

其核心思想是：为了能让 Flink 处理数据，buffers 需要处于可用状态。从前面的内容我们可以了解到，在 Flink 中，这些信息会被存储到一个个队列中，这些队列通过有限装载能力的 Buffer 池中。Buffer 池中的 Buffer 数据被消费之后将被回收。简而言之就是：从 Buffer 池中获取一个 Buffer，然后填满数据后，把 Buffer 放回到 Buffer 池中，消费者会从 Buffer 池中获取 Buffer 并使用里面的数据，然后再将 Buffer 放回 Buffer 池中。Buffer 池的大小在运行时中是动态的，取决于网络堆栈中的内存大小。Flink 程序会保证有足后的 buffer 数量以供使用，但是数据的消费速度取决于程序的处理速度。大内存可以保证程序能够承受短暂的反压（例如短期的峰值、或者短期的 GC 等），小内存则会对反压十分敏感。

### 基于 Credit 的反压机制

引入 Credit 表示下游 Task 的处理能力，使用 Backlog 表示上游 ResultPartition 中数据堆积的情况，通过 Credit 和 Backlog 的增减控制上下游数据生成和处理的频率。除了 Credit 和 Backlog 外，还有 ExclusiveBuffer 和 FloatingBuffer 队列，ExclusiveBuffer 是为每一个 InputChannel 申请专有 Buffer 队列，仅为当前 InputChannel 中的 Buffer 数据提供存储空间，FloatingBuffer 是一个浮动 Buffer 队列，所有 InputChannel 都可以向 NetworkBufferPool 申请 FloatingBuffer 资源，申请到的资源可以为当前 InputChannel 使用，但使用完毕后会立即释放。

RemoteInputChannel 启动过程中会去 NetworkBufferPool 中申请 ExclusiveBuffer 空间，具体大小可由用户参数配置。启动后会最终会将队列大小注册进 ResultPartition 中。上游产生新数据后，会增加 ResultSubPartition 中的 Backlog 值，Backlog 会跟随 Buffer 数据发送到 RemoteInputChannel 中。下游接收到信息后回解析出 Backlog 数据判断当前 RemoteInputChannel 是否有足够的空间，如果没有则会申请 FloatingBuffer 资源，并更新 unAnnounced Credit 。RemoteInputChannel 有足够可用 Buffer 时，会向 ResultPartition 发送 credit 信息。上游收到信息判断下游有足够多的 credit 后认为下游有足够的空间继续处理数据则会继续发送数据。

# 内存管理

## Flink 内存结构

[FLink 内存管理](https://flink.apache.org/2020/04/21/memory-management-improvements-with-apache-flink-1.10/)

![](https://flink.apache.org/img/blog/2020-04-21-memory-management-improvements-flink-1.10/total-process-memory.svg)

Flink Task Manager 进程是一个 JVM 进程，更高层面来说，它的内存包含 JVM Heap 和 Off-Heap 内存两部分。这些内存被 Flink 直接使用或者被 JVM 指定目的。

**请注意**，用户代码可以直接访问所有内存类型：JVM 堆、直接内存和本机内存。因此，Flink 无法真正控制其分配和使用。然而，有两种类型的 Off-Heap 由 Task 消耗并由 Flink 显式控制：

+ Off-Heap
+ Network Buffers

### 内存配置

| | option for taskmanager | option for JobManager |
| --- | :---: | :---: |
| Total Flink Memory | taskmanager.memory.flink.size | jobmanager.memory.flink.size |
| Total Process Memory | taksmanager.memory.process.size | jobmanager.memory.process.size |

+ 总进程内存：Flink Java 应用程序（包括用户代码）和JVM运行整个进程所消耗的总内存。
+ Flink 内存总量：仅 Flink Java 应用程序消耗的内存，包括用户代码，但不包括 JVM 为运行它分配的内存

## Flink 内存管理

1. Java 对象存储密度相对较低：对于常用的数据类型，例如 Boolean 类型数据占16字节内存空间，其中对象头占字节，Boolean 属性仅占1字节，其余7字节做对齐填充。而实际上仅1字节就能够代表 Boolean 值，这种情况造成了比较严重的内存空间浪费
2. Full GC 极大影响系统性能：使用 JVM 的垃圾回收机制对内存进行回收，在大数据量的情况下 GC 的性能会比较差，尤其对于大数据处理，有些数据对象处理完希望立即释放内存空间，但如果借助 JVM GC 自动回收，通常情况下会有秒级甚至分钟级别的延迟，这对系统的性能造成了非常大的影响
3. OutOfMemoryError 问题频发，严重影响系统稳定性：系统出现对象大小分配超过 JVM 内存限制时，就会触发 OutOfMemoryError，导致 JVM 宕机，影响整个数据处理进程。

鉴于以上 JVM 内存管理上的问题，在大数据领域已经有非常多的项目开始自己管理内存，目的就是让系统能够解决 JVM 内存管理的问题，以提升整个系统的性能和稳定性。只是独立内存管理会增加开发成本并提升系统的复杂度。

积极地内存管理，强调的是主动对内存资源进行管理。对 Flink 内存管理来讲，主要是将本来直接存储在堆内存上的数据对象，通过数据序列化处理，存储在预先分配的内存块上，该内存块也叫作 MemorySegment，代表了固定长度的内存范围，默认大小为 32KB，同时 MemorySegment 也是 Flink 的最小内存分配单元。

MemorySegment 将 JVM 堆内存和堆外内存进行集中管理，形成统一的内存访问视图。MemorySegment 提供了非常高效的内存读写方法，例如 getChar()、putChar() 等。如果 MemorySegment 底层使用的是 JVM 堆内存，数据通常会被存储至普通的字节数据（byte[]）中，如果 MemorySegment 底层使用的是堆外内存，则会借助 ByteBuffer 数据结构存储数据元素。基于 MemorySegment 内存块可以帮助 Flink 将数据处理对象尽可能连续地存储到内存中，且所有的数据对象都会序列化成二进制的数据格式，对一些 DBMS 风格的排序和连接算法来讲，这样能够将数据序列化和反序列化开销降到最低。 对于用户编写的自定义数据对象，例如 Person(String name, int age)，会通过高效的序列化工具将数据序列化成二进制数据格式，然后将二进制数据直接写入事先申请的内存块（MemorySegment）中，当再次需要获取数据的时候，通过反序列化工具将二进制数据格式转换成自定义对象。整个过程涉及的序列化和反序列化工具都已经在 Flink 内部实现，当然，Flink 也可以使用其他的序列化工具，例如 KryoSerializer 等。

在 MemorySegment 中如果因为内存空间不足，无法申请到更多的内存区域来存储对象时，Flink 会将MemorySegment 中的数据溢写到本地文件系统（SSD/Hdd）中。当再次需要操作数据时，会直接从磁盘中读取数据，保证系统不会因为内存不足而导致 OOM（Out Of Memory，超出内存空间），影响整个系统的稳定运行。

## 内存模型

### JVM Heap

JVM 堆内存分为 Framework 堆内存和 Task 堆内存，其中 Framework 堆内存用于 Flink 框架本身需要的内存空间。Task 内存则用于 Flink 算子以及用户代码，区别在于是否将内存计入 slot 计算资源中。

Off-Heap（非堆内存）主要包含了托管内存、直接内存以及 JVM 特定内存三部分

#### 托管内存

由 Flink 负责分配和管理本地（堆外）内存

#### 直接内存

分为 Framework 非堆内存、Task 非堆内存 和 Network。其中 Framework 非堆内存和 Task 非堆内存主要根据对外内存是否计入 Slot 资源进行区分。Network 内存存储空间主要用于基于 Netty 进行网络数据交换时，以 Network Buffer 的形式进行数据传输的本地缓冲

#### JVM 特定内存

不在 Flink 总内存范围内，包括 JVM 元空间和 JVM Overhead，JVM 元空间存储了 JVM 加载类的元数据，JVM Overhead 主要用于 JVM 开销，例如代码缓存、线程栈等。

对于 Flink 来讲，将内存划分成不同的区域，实现了更加精准地内存控制，并且可以通过 MemorySegmen 内存块的形式申请和管理内存，我们继续了解 MemorySegment 内存块的设计与实现。

## MemorySegment

在 Flink 中 MemorySegment 作为抽象类，分别被 HybridMemorySegment 和 HeapMemorySegment 继承和实现，HeapMemorySegment 提供了操作堆内存的方法，HybridMemorySegment 中提供了创建和操作堆内存和堆外内存的方法。在目前的 Flink 版本中，主要使用 HybridMemorySegment 处理堆与堆外内存，不再使用 HeapMemorySegment。

1. 为了尽可能避免直接实例化 HybridMemorySegment 对象，Flink 通过 MemorySegmentFactory 工厂类创建了 HybridMemorySegment。这是因为使用工厂模式控制类的创建，能够帮助 JIT 执行虚化（de-virtualized）和内联（inlined）的性能优化
2. DataOutputView 接口扩展了 java.io.DataOutput 接口，提供了对一个或多个 MemorySegment 执行写入操作的视图。使用 DataOutputView 提供的方法可以灵活高效地将数据按顺序写入连续的 MemorySegment 内存块中
3. DataInputView 接口扩展了 java.io.DataInput 接口，提供了对一个或多个 MemorySegment 执行读取操作的视图。使用 DataInputView 提供的方法可以灵活高效地按顺序读取 MemorySegment 中的内存数据
4. MemoryManager 主要用于管理排序、哈希和缓存等操作对应的内存空间，且这些操作主要集中在离线计算场景中
5. NetworkBufferPool 通过 MemorySegmentFactory 申请用于存储 NetworkBuffer 的 MemorySegment 内存空间。

在早期的 Flink 版本中，MemorySegment 是一个独立的 Final 类，没有区分 HeapMemorySegment 和 HybridMemorySegment 实现类，且仅支持管理堆内存。Flink 后期为了增加对堆外内存的支持，将 MemorySegment 类进行抽象，并引入了 HybridMemorySegment 类。这么做的主要目的是对 MemorySegment 方法进行去虚化和内联处理，从而更好地进行 JIT 编译优化。MemorySegment 是系统最底层的内存管理单元，在整个系统中的使用频率是非常高的。在 JIT 编译过程中，最好的处理方式就是明确需要调用的方法，早期 MemorySegment 因为是独立的 Final 类，JIT 编译时要调用的方法都是确定的。但如果分别将 HybridMemorySegment 和 HeapMemorySegment 两个子类加载到 JVM，此时 JIT 编译器只有在真正执行方法的时候才会确认是哪一个子类的方法，这样就无法提前判断仅有一个实现的虚方法调用，并把这些仅有一个实现的虚方法调用替换为唯一实现的直接调用，就会影响 JVM 的性能。

### 堆内内存管理

在 MemorySegmentFactory.wrap() 方法中可以直接将 byte[] buffer 数组封装成 MemorySegment，其中 byte[] 数组中的内存空间实际上就是从堆内存中申请的。除了将已有的 byte[] 数组空间转换成 MemorySegment 之外，在 MemorySegmentFactory 中同时提供了通过分配堆内存空间创建 MemorySegment 的方法。在 MemorySegmentFactory.allocateUnpooledSegment() 方法中通过指定参数 size 申请固定数量的 byte[] 数组，这里 new byte[size] 的操作实际上就是从堆内存申请内存空间。在 HybridMemorySegment 构造器中直接调用 MemorySegment 构造器，将 byte[] 数组赋值给 MemorySegment 中的 byte[] heapMemory 成员变量，并设定 offHeapBuffer 和 cleaner 为空。offHeapBuffer 和 cleaner 主要在 OffHeap 中使用，owner 参数表示当前的所有者，通常情况下设定为空。在 MemorySegment 的构造方法中提供了对 byte[] buffer 堆内存进行初始化的逻辑，在方法中首先将 buffer 赋值给 heapMemory，然后将 address 设定为 BYTE_ARRAY_BASE_OFFSET，表示 byte[] 数组内容的起始部分，然后根据数组对象和偏移量获取元素值（getObject）。

### 堆外内存管理

在 MemorySegment 中通过 ByteBuffer.allocateDirect(numBytes) 方法申请堆外内存，然后用 sun.misc.Unsafe 对象操作堆外内存。在 MemorySegmentFactory.allocateOffHeapUnsafeMemory() 方法中，调用 MemoryUtils.allocateUnsafe(size) 方法获取堆外内存空间的地址，然后调用 MemoryUtils.wrapUnsafeMemoryWithByteBuffer() 方法从给定的内存地址中申请内存空间，并转换成 ByteBuffer，最后通过 HybridMemorySegment 对象封装 ByteBuffer，并返回给使用方进行使用。

## DataInputView 与 DataOutputView

MemorySegment 解决了内存分块存储的问题，但如果需要使用连续的 MemorySegment 存储数据，就要借助 DataInputView 和 DataOutputView 组件实现。DataInputView 和 DataOutputView 中定义了一组 MemorySegment 的视图，其中 DataInputView 用于按顺序读取内存的数据，DataOutputView 用于按顺序将指定数据写入 MemorySegment。  
DataInputView 和 DataOutputView 分别继承了 java.io.DataInput 和 java.io.DataOut 接口，其中 DataInput 接口用于从二进制流中读取字节，且重构成所有 Java 基本类型数据。DataOutput 接口用于将任意 Java 基本类型转换为一系列字节，并将这些字节写入二进制流。

## 数据序列化与反序列化

Flink 的数据类型主要通过 TypeInformation 管理和定义，TypeInformation 也是 Flink 的类型系统核心类。用户自定义函数输入或返回值的类型信息都是通过 TypeInformation 实现的， TypeInformation 也充当了生成序列化器、比较器以及执行语义检查（例如是否存在用作 Join/grouping 主键字段）的工具。

TypeInformation 根据数据类型不同主要分为以下几种实现类型。

1. BasicTypeInfo：用于所有 Java 基础类型以及 String、Date、Void、BigInteger、BigDecimal 等类型
2. BasicArrayTypeInfo：用于由 Java 基础类型及 String 构成的数组类型
3. CompositeType：复合类型数据，例如 Java Tuple 类型对应 TupleTypeInfo、用户自定义 Pojo 类对应 PojoTypeInfo
4. WritableTypeInfo：用于支持扩展 Hadoop Writable 接口的数据类型
5. GenericTypeInfo：用于泛型类型数据。

# CDC 原理与应用

## Physical Replication
>
> PostgreSQL also supports physical replication, which streams raw block data rather than logical events on changes.
>

## PostgreSQL Logical Replication
>
> Logical decoding is the process of extracting all persistent changes to a database's tables into a coherent, easy to understand format which can be interpreted without detailed knowledge of the database's internal state.
>
> In PostgreSQL, logical decoding is implemented by decoding the contents of the write-ahead log, which describe changes on a storage level, into an application-specific form such as a stream of tuples or SQL statements.
>

logical replicaiton 是将所有数据库表的永久性改变(INSERT, UPDATE, DELETE)放入一个连续的、易读的、不需要理解数据库内部细节的数据形式。PostgreSQL 中，logical replication 是通过解码描述了存储层面上变动的 WAL 内容，将其应用在一个应用程序特定的形式，例如 tuple 流或者 SQL。

### Debezium
>
> Debezium is an open source distributed platform for change data capture. Start it up, point it at your databases, and your apps can start responding to all of the inserts, updates, and deletes that other apps commit to your databases. Debezium is durable and fast, so your apps can respond quickly and never miss an event, even when things go wrong.
>
> Debezium is built on top of Apache Kafka and provides a set of Kafka Connect compatible connectors. Each of the connectors works with a specific database management system (DBMS). Connectors record the history of data changes in the DBMS by detecting changes as they occur, and streaming a record of each change event to a Kafka topic. Consuming applications can then read the resulting event records from the Kafka topic.
>

Debezium 是一个第三方服务用于获取各种数据库的 CDC，支持诸如 MySQL, MongoDB, PostgreSQL, Jdbc 等。主要是基于 Kafka 服务，通过将数据库 CDC 数据发送给 Kafka，消费者再从 Kafka 消费。

### Decoder Output Plugin

+ decoderbufs：基于 Protobuf，由 Debezium 社区维护
+ pgoutput：PostgreSQL 10+ 版本之后由 PostgreSQL 社区提供的标准 logical decoding output plugin。

PostgreSQL通常会在一段时间后清除预写日志（WAL）段。这意味着 connector 没有对数据库所做的所有更改的完整历史记录。因此，当PostgreSQL连接器第一次连接到特定的PostgreSQL数据库时，它首先对每个数据库模式执行一致的快照。连接器完成快照后，将从创建快照的确切点开始继续流式传输更改。这样，连接器从所有数据的一致视图开始，并且不会省略在拍摄快照时所做的任何更改。

### Streaming Changes
>
> The PostgreSQL connector typically spends the vast majority of its time streaming changes from the PostgreSQL server to which it is connected. This mechanism relies on [PostgreSQL’s replication protocol](https://www.postgresql.org/docs/current/protocol-replication.html). This protocol enables clients to receive changes from the server as they are committed in the server’s transaction log at certain positions, which are referred to as Log Sequence Numbers (LSNs).
>

PostgreSQL 的 Streaming Replication Protocol 是实现物理复制以及逻辑复制的核心机制。

每当服务器提交事务时，一个单独的服务器进程就会调用逻辑解码插件中的回调函数。此函数处理事务中的更改，将其转换为特定格式（在Debezium插件的情况下为Protobuf或JSON），并将其写入输出流，然后客户端可以使用该输出流。

## Flink PostgreSQL CDC Connector

Flink CDC 在工作时会启用两个 worker，一个 worker 会拉取数据库的 streaming data 然后将数据存进 Handover 中，另一个 worker 则会从 Handover 中消费数据。

> The reason why don't use one workers is because debezium has different behaviours in snapshot phase and streaming phase.
>

在 Flink 工作流程中，会定期生成 Checkpoint 数据并保存到持久化文件中，在这个阶段时会影响流数据的处理，因此在中间加入 handover 层，使得在 Checkpoint 阶段，数据处理暂停，未来得及处理的数据将会暂存在 handover 内，并且因为 handover 阻塞，新的数据不会被拉取过来。等 Checkpoint 作业结束后，consumer 可以继续消费 handover 内的数据，新的数据也会被拉取到 cdc connector 内。

> The Postgres CDC source can’t work in parallel reading, because there is only one task can receive binlog events.
>

### 实现逻辑

CDC 通过工厂模式构建 DebeziumSourceFunction，而 DebeziumSourceFunction 则是继承了 RichSourceFunciton 并且实现了 Checkpoint 相关接口的功能。仔细查阅相关接口内容不难发现，继承 Checkpoint 相关接口狗，便可以在需要 Checkpoint 时保存一个快照

> This method is called when a snapshot for a checkpoint is requested. This acts as a hook to the function to ensure that all state is exposed by means previously offered through {[@link](/link ) FunctionInitializationContext} when the Function was initialized, or offered now by {[@link](/link ) FunctionSnapshotContext} itself.
>

在 DebeziumSourceFunction 启动前会调用 open() 方法，并建立一个 debezium-engine 线程，一个 handover 以及一个 consumer。handover 类似于一个大小为 1 的堆栈。当 cdc 产生一个数据时，再放进 handover 前先要判断 handover 内是否有未被获取的数据，如果有，则等待数据被消费。同样，consumer 在获取数据时如果 handover 内没有数据，则进入等待状态，直到有新数据放进去为止。

在启动时调用 run() 方法，会先获取数据库 replication slot，如果没有则创建。再去获取发布表情况，如果没有则全数据库表发布。接着去获取 Checkpoint 内 offset 情况，并根据获取内容进行恢复，如果没有则用当前最新数据。新建 DebeziumChangeFetcher 用于消费 Handover 内的数据，并根据配置异步启动 debezium-engine。

另外 debezium-engine 启动后，会定期（会有一个 `heartbeat.interval.ms`的配置）去数据库刷新 lsn 数据，以免在非活跃时期未拉取数据导致数据库 WAL 堆积。
