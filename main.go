package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/nullxjx/triton-service-go/models/vllm"
)

func generateRandomStr(length int) string {
	rand.Seed(time.Now().UnixNano())

	// 26个字母和10个数字
	characters := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	var randomStr string
	for i := 0; i < length; i++ {
		index := rand.Intn(len(characters))
		randomStr += string(characters[index])
	}

	return randomStr
}

func inferVllmInTriton() {
	model, err := vllm.NewModel(&vllm.ServerInfo{
		ServerIp: "81.69.152.80",
		GrpcPort: 8201,
	})
	if err != nil {
		fmt.Printf("Model init err: %v\n", err)
		return
	}

	params := &vllm.InferParams{
		Prompt:         "def quick_sort():",
		ModelName:      "triton-vllm-code-llama-model",
		ModelVersion:   "1",
		TimeoutSeconds: 100,
		InferConfig: &vllm.InferConfig{
			StopWords:   []string{},
			MaxTokens:   32,
			Temperature: 0,
			BeamWidth:   4,
			TopP:        1,
		},
	}
	times := 1
	var totalTimeSpent int64 = 0
	totalTokens := 0
	for i := 0; i < times; i++ {
		output, err := vllm.Infer(params, model)
		if err != nil {
			fmt.Printf("Infer failed, err: %v\n", err)
			return
		}
		totalTimeSpent += model.GetDetailedInferResult().TimeSpent
		totalTokens += model.GetDetailedInferResult().OutputTokens[0][0]
		fmt.Printf("output: %#v\n", output)
		time.Sleep(1 * time.Second)
	}

	fmt.Printf("output tokens : %v, time spent: %.3f s", float64(totalTokens)/float64(times),
		float64(totalTimeSpent)/float64(times)/1000)
}

func findTargetPrompt() {
	model, err := vllm.NewModel(&vllm.ServerInfo{
		ServerIp: "110.40.186.207",
		GrpcPort: 8701,
	})
	if err != nil {
		fmt.Printf("Model init err: %v\n", err)
		return
	}

	for i := 0; i < 100; i++ {
		prompt := generateRandomStr(500)
		params := &vllm.InferParams{
			Prompt:         prompt,
			ModelName:      "vllm",
			ModelVersion:   "1",
			TimeoutSeconds: 100,
			InferConfig: &vllm.InferConfig{
				StopWords:   []string{},
				MaxTokens:   1024,
				Temperature: 0,
				BeamWidth:   1,
				TopP:        1,
			},
		}

		_, err = vllm.Infer(params, model)
		if err != nil {
			fmt.Printf("Infer failed, err: %v\n", err)
			return
		}

		maxTokens := model.GetDetailedInferResult().OutputTokens[0][0]
		if maxTokens >= 1024 {
			fmt.Printf("prompt: %v, maxTokens: %v", prompt, maxTokens)
			return
		}
	}
}

func main() {
	//quickInfer()
	inferVllmInTriton()
	//findTargetPrompt()
}
