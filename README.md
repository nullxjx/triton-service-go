# triton-service-go
Unofficial golang package for the Triton Inference Server(https://github.com/triton-inference-server/server)
Triton Inference Server - Golang API

---

### Attention

* Currently only supported up to version 21.05 (Triton Inference Server), but compatible with 9x% of 22.07 versions

---

### Feature

* Support HTTP/GRPC
* Easy to use it
* Maybe High Performance
* Implement 90% API of Triton Inference Server HTTP/GRPC Protocol

--- 

### Usage

* Download
```shell
go get -u github.com/sunhailin-Leo/triton-service-go
```

---

### Version

* version 1.1.7
  * use `bytedance/sonic` instead of `encoding/json`
  * use `errors.New` instead of `fmt.Errorf`
  * remove `fmt` package usage
  * update `go.mod`

* version 1.1.2
  * update go.mod

* version 1.0.0
    * Implement about 90% API of Triton Inference Server HTTP/GRPC Protocol