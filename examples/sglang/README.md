<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LLM Deployment Examples using SGLang

This directory contains examples and reference implementations for deploying Large Language Models (LLMs) in various configurations using SGLang. SGLang internally uses ZMQ to communicate between the ingress and the engine processes. For Dynamo, we leverage the runtime to communicate directly with the engine processes and handle ingress and pre/post processing on our end.

## Use the Latest Release

We recommend using the latest stable release of dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

## Deployment Architectures

See [deployment architectures](../llm/README.md#deployment-architectures) to learn about the general idea of the architecture. SGLang currently supports aggregated and disaggregated serving. KV routing support is coming soon!

## Getting Started

1. Choose a deployment architecture based on your requirements
2. Configure the components as needed
3. Deploy using the provided scripts

### Prerequisites

Start required services (etcd and NATS) using [Docker Compose](../../deploy/metrics/docker-compose.yml)

```bash
docker compose -f deploy/metrics/docker-compose.yml up -d
```

### Build docker

```bash
# On an x86 machine - sglang does not support ARM yet
./container/build.sh --framework sglang
```

### Run container

```bash
./container/run.sh -it --framework sglang
```

### Example architectures

#### Aggregated

```bash
cd /workspace/examples/sglang
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml
```

#### Aggregated with router

> [!NOTE]
> The current implementation of `examples/sglang/components/worker.py` publishes _placeholder_ engine metrics to keep the Dynamo KV-router happy. Real-time metrics will be surfaced directly from the SGLang engine once the following pull requests are merged:
> • Upstream: [sgl-project/sglang #6721](https://github.com/sgl-project/sglang/pull/6721) – _Expose runtime KV-cache & request metrics_.
> • Dynamo: [ai-dynamo/dynamo #1465](https://github.com/ai-dynamo/dynamo/pull/1465) – _feat: receive kvmetrics from sglang scheduler_.
>
> After these are in, the TODOs in `worker.py` will be resolved and the placeholder logic removed.

```bash
cd /workspace/examples/sglang
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml --Frontend.router=kv
```

#### Disaggregated

<details>
<summary>SGLang Load Balancer vs Dynamo Discovery</summary>

SGLang uses a mini load balancer to route requests to handle disaggregated serving. The load balancer functions as follows:

1. The load balancer receives a request from the client
2. A random `(prefill, decode)` pair is selected from the pool of available workers
3. Request is sent to both `prefill` and `decode` workers via asyncio tasks
4. Internally disaggregation is done from prefill -> decode

Because Dynamo has a discovery mechanism, we do not use a load balancer. Instead, we first route to a random prefill worker, select a random decode worker, and then send the request to both. Internally, SGLang's bootstrap server (which is a part of the `tokenizer_manager`) is used in conjuction with NIXL to handle the kv transfer.

</details>

> [!IMPORTANT]
> Disaggregated serving in SGLang currently requires each worker to have the same tensor parallel size [unless you are using an MLA based model](https://github.com/sgl-project/sglang/pull/5922)

```bash
cd /workspace/examples/sglang
dynamo serve graphs.disagg:Frontend -f ./configs/disagg.yaml
```

#### Embedding Models

We support deploying embedding models via SGLang as well! Here's an example that uses the [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) model.

```bash
cd /workspace/examples/sglang
dynamo serve graphs.embedding:Frontend -f ./configs/embedding.yaml
```

##### Disaggregated with MoE models and DP attention

SGLang also supports DP attention for MoE models. We provide an example config for this in `configs/disagg-dp-attention.yaml` which is based on the [DeepSeek-R1-Small-2layers](https://huggingface.co/silence09/DeepSeek-R1-Small-2layers) model. You can use this configuration to test out disaggregated serving on a single node before scaling to the full DeepSeek-R1 model across multiple nodes.

```bash
# note this will require 4 GPUs
cd /workspace/examples/sglang
dynamo serve graphs.disagg:Frontend -f ./configs/disagg-dp-attention.yaml
```

In order to scale to the full DeepSeek-R1 model, you can follow the instructions in the [multinode-examples.md](./multinode-examples.md) file.

##### Disaggregated with WideEP

Dynamo supports SGLang's implementation of wide expert parallelism and large scale P/D for DeepSeek-R1! You can find detailed deployment and benchmarking instructions [here](./dsr1-wideep.md)
