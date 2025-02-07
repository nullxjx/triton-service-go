package test

import (
	"errors"
	"testing"

	"github.com/nullxjx/triton-service-go/triton"
	"github.com/valyala/fasthttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func TestTritonHTTPClientInit(_ *testing.T) {
	client := &fasthttp.Client{}
	trtClient := triton.NewTritonClientWithOnlyHTTP("127.0.0.1:9001", client)
	if trtClient == nil {
		panic(errors.New("init triton http client failed"))
	}
}

func TestTritonGRPCClientInit(_ *testing.T) {
	_, err := grpc.Dial("127.0.0.1:9000", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		panic(err)
	}
}
