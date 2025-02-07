package vllm

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"strconv"
	"time"

	"github.com/nullxjx/triton-service-go/triton"
	"github.com/nullxjx/triton-service-go/utils"

	"github.com/google/uuid"
	"google.golang.org/grpc"
)

type InferConfig struct {
	StopWords   []string
	MaxTokens   uint32
	TopK        uint32
	TopP        float32
	BeamWidth   uint32
	Temperature float32
}

type DetailedInferResult struct {
	TimeSpent      int64   // 所花时间，单位毫秒
	InputLen       []int   // 输入字符串长度
	InputTokens    []int   // 输入token数目
	OutputLen      [][]int // 输出字符串长度，由于每条输入都可能生成多条输出，所以这里是一个二维数组
	OutputTokens   [][]int // 输出token数目
	MaxInputTokens int     // 所有输入中最长的token数目
}

type ModelService struct {
	isGRPC        bool
	maxSeqLength  int
	modelName     string
	tritonService *triton.TritonClientService
	inferCallback triton.DecoderFunc
	inferConfig   *InferConfig
	inferResult   DetailedInferResult
}

// inputIDsTensor 对于 python_backend 来说，需要对 prompt 的 bytes 数组塞入长度信息到最前面
// https://github.com/triton-inference-server/python_backend/blob/4c4a552b047ff00ca8c6b87ba1fe4ac8f83eaf24/src/resources/triton_python_backend_utils.py#L72C17-L72C17
func inputIDsTensor(prompt []byte, name, datatype string, lenOfEveryArray int) (
	*triton.ModelInferRequest_InferInputTensor, []byte, error) {
	var serializedTensor []byte
	sizeBytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(sizeBytes, uint32(lenOfEveryArray))
	serializedTensor = append(serializedTensor, sizeBytes...)
	serializedTensor = append(serializedTensor, prompt...)
	return &triton.ModelInferRequest_InferInputTensor{
		Name:     name,
		Datatype: datatype,
		Shape:    []int64{1},
	}, serializedTensor, nil
}

func streamTensor(isStream []bool) (*triton.ModelInferRequest_InferInputTensor, []byte) {
	return &triton.ModelInferRequest_InferInputTensor{
		Name:     "STREAM",
		Datatype: "BOOL",
		Shape:    []int64{int64(1)},
	}, utils.SliceToBytes(isStream)
}

func prepareInputTensors(prompt []byte, params []byte, stream bool) (
	[]*triton.ModelInferRequest_InferInputTensor, [][]byte, error) {
	iiTS, iiBS, err := inputIDsTensor(prompt, "PROMPT", "BYTES", len(prompt))
	if err != nil {
		return nil, nil, err
	}
	paramsTS, paramsBS, err := inputIDsTensor(params, "SAMPLING_PARAMETERS", "BYTES", len(params))
	if err != nil {
		return nil, nil, err
	}
	irlpTS, irlpBS := streamTensor([]bool{stream})
	return []*triton.ModelInferRequest_InferInputTensor{
			iiTS,
			irlpTS,
			paramsTS,
		},
		[][]byte{
			iiBS,
			irlpBS,
			paramsBS,
		}, nil
}

func (m *ModelService) SetInferConfig(config *InferConfig) {
	m.inferConfig = config
}

func (m *ModelService) GetDetailedInferResult() DetailedInferResult {
	return m.inferResult
}

func (m *ModelService) prepareParams() map[string]*triton.InferParameter {
	params := make(map[string]*triton.InferParameter)
	stopJson, _ := json.Marshal(m.inferConfig.StopWords)
	params["temperature"] = &triton.InferParameter{
		ParameterChoice: &triton.InferParameter_StringParam{StringParam: strconv.FormatFloat(
			float64(m.inferConfig.Temperature), 'f', -1, 32)},
	}
	params["top_p"] = &triton.InferParameter{
		ParameterChoice: &triton.InferParameter_StringParam{StringParam: strconv.FormatFloat(
			float64(m.inferConfig.TopP), 'f', -1, 32)},
	}
	params["presence_penalty"] = &triton.InferParameter{
		ParameterChoice: &triton.InferParameter_StringParam{StringParam: strconv.FormatFloat(
			float64(PresencePenalty), 'f', -1, 32)},
	}
	params["frequency_penalty"] = &triton.InferParameter{
		ParameterChoice: &triton.InferParameter_StringParam{StringParam: strconv.FormatFloat(
			float64(FrequencyPenalty), 'f', -1, 32)},
	}
	params["stop"] = &triton.InferParameter{
		ParameterChoice: &triton.InferParameter_StringParam{StringParam: string(stopJson)},
	}
	params["n"] = &triton.InferParameter{
		ParameterChoice: &triton.InferParameter_Int64Param{Int64Param: int64(m.inferConfig.BeamWidth)},
	}
	params["best_of"] = &triton.InferParameter{
		ParameterChoice: &triton.InferParameter_Int64Param{Int64Param: int64(m.inferConfig.BeamWidth)},
	}
	params["max_tokens"] = &triton.InferParameter{
		ParameterChoice: &triton.InferParameter_Int64Param{Int64Param: int64(m.inferConfig.MaxTokens)},
	}
	params["logprobs"] = &triton.InferParameter{
		ParameterChoice: &triton.InferParameter_Int64Param{Int64Param: int64(1)},
	}
	useBeamSearch := false
	if m.inferConfig.BeamWidth > 1 {
		useBeamSearch = true
	}
	params["use_beam_search"] = &triton.InferParameter{
		ParameterChoice: &triton.InferParameter_BoolParam{BoolParam: useBeamSearch},
	}
	return params
}

// ModelInfer API to call Triton Inference Server.
func (m *ModelService) ModelInfer(
	prompt string,
	modelName, modelVersion string,
	requestTimeout time.Duration,
) ([]interface{}, error) {
	useBeamSearch := false
	if m.inferConfig.BeamWidth > 1 {
		useBeamSearch = true
	}
	params := &Params{
		Stop:          m.inferConfig.StopWords,
		MaxTokens:     int32(m.inferConfig.MaxTokens),
		Temperature:   m.inferConfig.Temperature,
		N:             int32(m.inferConfig.BeamWidth),
		UseBeamSearch: useBeamSearch,
		TopP:          m.inferConfig.TopP,
		BestOf:        int32(m.inferConfig.BeamWidth),
	}
	paramsBytes, _ := json.Marshal(params)
	tensors, rawInputs, err := prepareInputTensors([]byte(prompt), paramsBytes, false)
	if err != nil {
		return nil, errors.New("prepare tensor failed")
	}
	id := uuid.New().String()
	return m.tritonService.ModelGRPCStreamInfer(
		id, tensors, rawInputs, m.prepareParams(),
		modelName, modelVersion, requestTimeout,
		m.inferCallback, m,
	)
}

func NewModelService(
	grpcConn *grpc.ClientConn,
	modelInferCallback triton.DecoderFunc,
) (*ModelService, error) {
	srv := &ModelService{
		maxSeqLength:  DefaultMaxSeqLength,
		tritonService: triton.NewTritonClientForAll("", nil, grpcConn),
		inferCallback: modelInferCallback,
		inferResult: DetailedInferResult{
			TimeSpent:      0,
			InputLen:       nil,
			InputTokens:    nil,
			OutputLen:      nil,
			OutputTokens:   nil,
			MaxInputTokens: 0,
		},
	}
	return srv, nil
}
