package main

import (
	"aqsml/internal/aqsml"
	"flag"
	"fmt"
	"os"
	"strings"
)

func main() {
	cfg := aqsml.DefaultConfig()

	inputPath := flag.String("input", cfg.InputPath, "Ruta CSV de entrada (clean o final)")
	targetCol := flag.String("target", cfg.TargetCol, "Columna target")
	numFeaturesRaw := flag.String("num-features", strings.Join(cfg.NumericFeatures, ","), "Columnas numericas separadas por coma")
	catFeaturesRaw := flag.String("cat-features", strings.Join(cfg.CatFeatures, ","), "Columnas categoricas separadas por coma")
	ordFeaturesRaw := flag.String("ord-features", strings.Join(cfg.OrdFeatures, ","), "Subconjunto de categoricas tratadas como ordinales")
	maxCatLevels := flag.Int("max-cat-levels", cfg.MaxCatLevels, "Maximo de niveles por columna categorica para codificar")
	trainYearEnd := flag.Int("train-year-end", cfg.TrainYearEnd, "Anios <= este valor son train; mayores son test")
	usePCA := flag.Bool("use-pca", cfg.UsePCA, "Aplicar PCA antes de entrenar")
	pcaVarianceGoal := flag.Float64("pca-variance", cfg.PCAVarianceGoal, "Varianza explicada acumulada objetivo para PCA")
	maxRows := flag.Int("max-rows", cfg.MaxRows, "Maximo de filas a cargar (0 = sin limite)")
	parserWorkers := flag.Int("parser-workers", cfg.ParserWorkers, "Numero de workers para parseo/validacion")
	encoderWorkers := flag.Int("encoder-workers", cfg.EncoderWorkers, "Numero de workers para encoding de filas")
	rawBuffer := flag.Int("raw-buffer", cfg.RawBuffer, "Buffer de canal rawRows")
	parsedBuffer := flag.Int("parsed-buffer", cfg.ParsedBuffer, "Buffer de canal parsedRows")
	encodedBuffer := flag.Int("encoded-buffer", cfg.EncodedBuffer, "Buffer de canal encodedRows")
	profile := flag.Bool("profile", cfg.Profile, "Imprime tiempos y memoria por etapa")
	solver := flag.String("solver", cfg.Solver, "Solver de regresion: ridge, normal o svd")
	fitWorkers := flag.Int("fit-workers", cfg.FitWorkers, "Workers para acumulacion concurrente de XTX/XTy")
	ridgeLambda := flag.Float64("ridge-lambda", cfg.RidgeLambda, "Lambda de regularizacion Ridge")
	flag.Parse()

	cfg.InputPath = *inputPath
	cfg.TargetCol = *targetCol
	cfg.NumericFeatures = aqsml.SplitAndTrim(*numFeaturesRaw)
	cfg.CatFeatures = aqsml.SplitAndTrim(*catFeaturesRaw)
	cfg.OrdFeatures = aqsml.SplitAndTrim(*ordFeaturesRaw)
	cfg.MaxCatLevels = *maxCatLevels
	cfg.TrainYearEnd = *trainYearEnd
	cfg.UsePCA = *usePCA
	cfg.PCAVarianceGoal = *pcaVarianceGoal
	cfg.MaxRows = *maxRows
	cfg.ParserWorkers = *parserWorkers
	cfg.EncoderWorkers = *encoderWorkers
	cfg.RawBuffer = *rawBuffer
	cfg.ParsedBuffer = *parsedBuffer
	cfg.EncodedBuffer = *encodedBuffer
	cfg.Profile = *profile
	cfg.Solver = *solver
	cfg.FitWorkers = *fitWorkers
	cfg.RidgeLambda = *ridgeLambda

	if err := aqsml.Run(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
