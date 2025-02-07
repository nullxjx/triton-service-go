package vllm

const (
	RequestLen          = 512
	TopK                = 1
	TopP                = 0.1
	BeamRate            = 0
	Temperature         = 0.0
	LenPenalty          = 1.0
	RepetitionPenalty   = 1.0
	PresencePenalty     = 1.0
	FrequencyPenalty    = 1.0
	RandomSeed          = 0
	NeedReturnLogProbs  = false
	BeamWidth           = 1
	NeedStartID         = 1
	NeedEndID           = 2
	DefaultMaxSeqLength = 48
)
