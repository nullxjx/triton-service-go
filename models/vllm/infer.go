package vllm

import (
	"errors"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type ServerInfo struct {
	ServerIp string
	GrpcPort int
}

type InferParams struct {
	Prompt         string
	ModelName      string
	ModelVersion   string
	TimeoutSeconds int
	InferConfig    *InferConfig
}

func NewModel(info *ServerInfo) (*ModelService, error) {
	grpcAddr := fmt.Sprintf("%s:%d", info.ServerIp, info.GrpcPort)
	grpcClient, grpcErr := grpc.Dial(grpcAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if grpcErr != nil {
		return nil, errors.New("init grpc client error")
	}

	// Service
	modelService, initErr := NewModelService(grpcClient, InferCallback)
	if initErr != nil {
		return nil, errors.New(fmt.Sprintf("init model error: %v", initErr))
	}
	return modelService, nil
}

func Infer(params *InferParams, model *ModelService) ([][]string, error) {
	if params.InferConfig == nil {
		// 用户没设置config时，使用默认的config
		model.SetInferConfig(&InferConfig{
			StopWords:   []string{},
			MaxTokens:   RequestLen,
			TopK:        TopK,
			TopP:        TopP,
			BeamWidth:   BeamWidth,
			Temperature: Temperature,
		})
	} else {
		model.SetInferConfig(params.InferConfig)
	}
	inferResult, inferErr := model.ModelInfer(
		params.Prompt,
		params.ModelName,
		params.ModelVersion,
		time.Duration(params.TimeoutSeconds)*time.Second,
	)
	if inferErr != nil {
		return nil, errors.New(fmt.Sprintf("infer error: %v", inferErr))
	}

	stringSlice := make([][]string, len(inferResult))
	for i, v := range inferResult {
		stringSlice[i] = v.([]string)
	}
	return stringSlice, nil
}
