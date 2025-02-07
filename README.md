fork from [triton-service-go](https://github.com/sunhailin-Leo/triton-service-go)，添加了对 fastertransformer 模型的支持

## feature
- 支持 http / grpc 调用
- 支持 batch 化调用
- 支持 stopWords

## usage
然后在go.mod中加入以下依赖，目前稳定版是 v1.2.6
```bash
require (
	github.com/nullxjx/triton-service-go v1.2.6
)
```
示例代码见: [main.go](main.go)

注意：
1. 调用完成返回的result是一个List，每个结果与输入prompt按顺序对应
2. result里面是模型原始输出，不做任何数据处理或者截断逻辑

## limitation
- 暂时不支持对一个batch的不同输入使用不同的stopWords
- llama模型使用 [sentencepiecego](https://github.com/activezhao/sentencepiecego) 来初始化tokenizer，这个库依赖一个动态链接文件

## history
- v1.1.0 第一个稳定版本
- v1.1.1 修改padding字符为空格
- v1.1.2 修复http报错过程出现空指针的bug
- v1.2.0
  - 允许设置StopWords MaxTokens TopK TopP BeamWidth Temperature推理参数
  - 输出结果适配 BeamWidth > 1 的情况
  - 优化调用传参
  - 代码结构重构和精简，提升逻辑复用程度
- v.1.2.4 支持triton+vllm模式部署的模型
- v.1.2.6 移除对fastertransformer调用的支持