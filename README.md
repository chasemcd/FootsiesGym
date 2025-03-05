# FootsiesGym

Implementation of the Footsies Unity game as a reinforcement learning environment.

## Installation

```bash
conda create -n footsiesgym python=3.10
conda activate footsiesgym
pip install -r requirements.txt
```



## Visualizing a Policy

To visualize gameplay, you'll need to run the windowed version of the game. This repository includes the windowed and headless Linux builds (TODO: add Windows/Mac windowed builds).
1. Unpack



---
###  gRPC / Protobuf Updates

You'll need to generate `Footsies.cs` and `FootsiesGrpc.cs`. This repo includes what you need on a __Windows__ machine:

```
.\protoc\bin\protoc.exe --csharp_out=.\env\game\proto\ --grpc_out=.\env\game\proto\ --plugin=protoc-gen-grpc=.\plugins\grpc_csharp_plugin.exe .\env\game\proto\footsies_service.proto
```

The corresponding python files are also necessary (make sure to `pip install grpcio-tools grpcio`):

```
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. .\env\game\proto\footsies_service.proto
```
