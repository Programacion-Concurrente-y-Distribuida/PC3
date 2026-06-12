package aqsml

import (
	"fmt"
	"os"
	"runtime"
	"strings"

	"gonum.org/v1/gonum/mat"
)

var (
	DefaultNumericFeatures = []string{
		"Latitude",
		"Longitude",
		"Year",
		"Observation Count",
		"Observation Percent",
		"Valid Day Count",
		"Required Day Count",
		"Exceptional Data Count",
		"Null Data Count",
		"Num Obs Below MDL",
		"Primary Exceedance Count",
		"Secondary Exceedance Count",
	}
	DefaultCategoricalFeatures = []string{
		"pollutant",
		"Parameter Code",
		"Sample Duration",
		"Event Type",
		"Units of Measure",
		"Pollutant Standard",
		"State Code",
	}
	DefaultOrdinalFeatures = []string{}
)

type Config struct {
	InputPath       string
	TargetCol       string
	NumericFeatures []string
	CatFeatures     []string
	OrdFeatures     []string
	MaxCatLevels    int
	TrainYearEnd    int
	UsePCA          bool
	PCAVarianceGoal float64
	MaxRows         int
	ParserWorkers   int
	EncoderWorkers  int
	RawBuffer       int
	ParsedBuffer    int
	EncodedBuffer   int
	Profile         bool
	Solver          string
	FitWorkers      int
	RidgeLambda     float64
}

type rowData struct {
	Year      int
	NumValues []float64
	CatValues []string
	Target    float64
}

type dataset struct {
	TrainX *mat.Dense
	TrainY *mat.Dense
	TestX  *mat.Dense
	TestY  *mat.Dense
}

type metrics struct {
	MAE  float64
	RMSE float64
	R2   float64
}

type nominalEncoder struct {
	ColName     string
	ColIdx      int
	BaseLevel   string
	ActiveLvls  []string
	LvlToOffset map[string]int
}

type ordinalEncoder struct {
	ColName   string
	ColIdx    int
	LevelRank map[string]float64
}

type catEncodingPlan struct {
	Nominal         []nominalEncoder
	Ordinal         []ordinalEncoder
	DroppedCols     []string
	FeatureNames    []string
	NominalColsUsed []string
	OrdinalColsUsed []string
}

func DefaultConfig() Config {
	cpus := runtime.NumCPU()
	// La lectura del CSV es serial y domina la carga; mas workers por pool
	// solo agregan contencion de canales (2/2 midio ~2x mas rapido que NumCPU/NumCPU÷2
	// sobre el dataset de 3M).
	parserWorkers := 2
	encoderWorkers := 2
	return Config{
		InputPath:       AutoInput(),
		TargetCol:       "Arithmetic Mean",
		NumericFeatures: append([]string(nil), DefaultNumericFeatures...),
		CatFeatures:     append([]string(nil), DefaultCategoricalFeatures...),
		OrdFeatures:     append([]string(nil), DefaultOrdinalFeatures...),
		MaxCatLevels:    80,
		TrainYearEnd:    2020,
		UsePCA:          false,
		PCAVarianceGoal: 0.95,
		MaxRows:         250000,
		ParserWorkers:   parserWorkers,
		EncoderWorkers:  encoderWorkers,
		RawBuffer:       parserWorkers * 2,
		ParsedBuffer:    encoderWorkers * 2,
		EncodedBuffer:   3,
		Profile:         false,
		Solver:          "ridge",
		// El barrido de fit-workers (scripts/worker_benchmark) midio el mejor
		// tiempo y la menor varianza con 2x los nucleos fisicos.
		FitWorkers:      cpus * 2,
		RidgeLambda:     1,
	}
}

func AutoInput() string {
	if _, err := os.Stat("aqs_final_3M.csv"); err == nil {
		return "aqs_final_3M.csv"
	}
	return "aqs_clean.csv"
}

func SplitAndTrim(s string) []string {
	if strings.TrimSpace(s) == "" {
		return []string{}
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		v := strings.TrimSpace(p)
		if v != "" {
			out = append(out, v)
		}
	}
	return out
}

func safeJoin(v []string) string {
	if len(v) == 0 {
		return "(ninguna)"
	}
	return strings.Join(v, ", ")
}

func exitf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
