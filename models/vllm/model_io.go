package vllm

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"

	"github.com/nullxjx/triton-service-go/triton"
)

var (
	completionOutputsRE = regexp.MustCompile(`index=(\d+), text=(.*?), token_ids=(\[[\d, ]+\]), cumulative_logprob=([-]?[\d.]+), logprobs=(\[.*?\]), finish_reason=(\w+)`)
)

// Params 是 vLLM 推理请求参数
type Params struct {
	Stop             []string `json:"stop"`              // 停止符
	MaxTokens        int32    `json:"max_tokens"`        // 最大生成长度
	Temperature      float32  `json:"temperature"`       // 温度，控制生成结果的多样性
	N                int32    `json:"n"`                 // 结果个数
	UseBeamSearch    bool     `json:"use_beam_search"`   // 是否使用束搜索
	TopP             float32  `json:"top_p"`             // 控制生成建议的多样性。较高的值 (接近 1) 会产生更多样化的建议，较低的值 (接近 0) 会产生更加确定性的建议。
	Logprobs         int32    `json:"logprobs"`          // 一个整数，表示返回与生成建议相关的概率分布信息
	Echo             bool     `json:"echo,omitempty"`    // 在返回的建议中包含输入的 prompt
	PresencePenalty  float32  `json:"presence_penalty"`  // 控制建议中重复内容的惩罚。较高的值会减少重复内容，较低的值可能会导致更多重复。
	FrequencyPenalty float32  `json:"frequency_penalty"` // 控制建议中罕见 tokens 的惩罚。较高的值会导致建议更倾向于使用常见 tokens，较低的值可能会导致建议包含更多罕见 tokens
	BestOf           int32    `json:"best_of"`           // 请求生成多个建议（由 n 参数控制），best_of 值越大，生成的建议质量可能越高，但计算成本也越高
	User             string   `json:"user,omitempty"`    // 用户信息
}

// LogProbInfo 概率分布信息
type LogProbInfo struct {
	Tokens        []string             `json:"tokens"`
	TokenLogprobs []float32            `json:"token_logprobs"`
	TopLogprobs   []map[string]float32 `json:"top_logprobs"`
	TextOffset    []int32              `json:"text_offset"`
}

// CompletionOutput 推理返回的结果
type CompletionOutput struct {
	Text              string      `json:"text"`
	Index             int         `json:"index"`
	FinishReason      string      `json:"finish_reason"`
	TokenIDs          []int32     `json:"token_ids"`
	CumulativeLogprob int32       `json:"cumulative_logprob"`
	Logprobs          LogProbInfo `json:"logprobs"`
}

// Choice 推理结果相关
type Choice struct {
	Text         string      `json:"text"`
	Index        int         `json:"index"`
	Logprobs     LogProbInfo `json:"logprobs"`
	FinishReason string      `json:"finish_reason"`
}

// Usage token 相关统计
type Usage struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

// InferRsp 推理结果基本信息
type InferRsp struct {
	Choices []*Choice `json:"choices"`
	Usage   *Usage    `json:"usage"`
}

// deserializeBytesTensor 对于 python_backend 来说，反序列化时，需要跳过前面的 4 个字节，因为长度位于前 4 个字节中，后跟内容
// https://github.com/triton-inference-server/python_backend/blob/4c4a552b047ff00ca8c6b87ba1fe4ac8f83eaf24/src/resources/triton_python_backend_utils.py#L95
func deserializeBytesTensor(encodedTensor []byte) ([]byte, error) {
	tensor := make([]byte, 0)
	offset := 0
	for offset < len(encodedTensor) {
		if offset+4 > len(encodedTensor) {
			return nil, fmt.Errorf("invalid encoded tensor format")
		}
		elementSize := binary.LittleEndian.Uint32(encodedTensor[offset : offset+4])
		offset += 4
		if offset+int(elementSize) > len(encodedTensor) {
			return nil, fmt.Errorf("invalid encoded tensor format")
		}
		elementBytes := encodedTensor[offset : offset+int(elementSize)]
		offset += int(elementSize)
		tensor = append(tensor, elementBytes...)
	}
	return tensor, nil
}

func InferCallback(inferResponse interface{}, params ...interface{}) ([]interface{}, error) {
	var m *ModelService
	for _, param := range params {
		switch v := param.(type) {
		case *ModelService:
			m = v
		}
	}
	if m == nil {
		return nil, errors.New("callback function need model info")
	}

	rsp, ok := inferResponse.(*triton.ModelInferResponse)
	if !ok {
		return nil, errors.New("grpc response 类型不是 *triton.ModelInferResponse 类型")
	}

	// 这里获取infer开始和结束时间
	// 根据调用链路，到这里为止，params一共包含3个参数，infer时间是第3个
	m.inferResult.TimeSpent = params[1].(int64)
	id := params[2].(string)
	if rsp.Id != id {
		return nil, fmt.Errorf("InferVLLMInTriton rsp not matches req, id:%v", id)
	}
	if len(rsp.RawOutputContents) == 0 {
		return nil, nil
	}

	outputs, err := deserializeBytesTensor(rsp.RawOutputContents[0])
	if err != nil {
		return nil, err
	}

	var rspData *InferRsp
	if err = json.Unmarshal(outputs, &rspData); err != nil {
		return nil, errors.New(fmt.Sprintf("unmarshal outputs error: %v", err))
	}

	var result []interface{}
	var resultPerBatch []string
	var outputTokens []int
	var outputLen []int
	var inputTokens []int
	for _, choice := range rspData.Choices {
		outputTokens = append(outputTokens, int(rspData.Usage.CompletionTokens))
		outputLen = append(outputLen, len(choice.Text))
		inputTokens = append(inputTokens, int(rspData.Usage.PromptTokens))
		resultPerBatch = append(resultPerBatch, choice.Text)
	}
	m.inferResult.OutputTokens = [][]int{outputTokens}
	m.inferResult.OutputLen = [][]int{outputLen}
	m.inferResult.InputTokens = inputTokens
	result = append(result, resultPerBatch)

	return result, nil
}
