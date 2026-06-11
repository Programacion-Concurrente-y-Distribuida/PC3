package aqsml

import (
	"fmt"
	"strings"
)

func Run(cfg Config) error {
	if len(cfg.NumericFeatures) == 0 && len(cfg.CatFeatures) == 0 {
		return fmt.Errorf("debes indicar al menos una columna en num-features o cat-features")
	}
	if cfg.PCAVarianceGoal <= 0 || cfg.PCAVarianceGoal > 1 {
		return fmt.Errorf("pca-variance debe estar entre (0,1]")
	}
	if cfg.MaxCatLevels < 2 {
		return fmt.Errorf("max-cat-levels debe ser >= 2")
	}
	if cfg.ParserWorkers < 1 || cfg.EncoderWorkers < 1 {
		return fmt.Errorf("parser-workers y encoder-workers deben ser >= 1")
	}
	if cfg.RawBuffer < 1 || cfg.ParsedBuffer < 1 || cfg.EncodedBuffer < 1 {
		return fmt.Errorf("raw-buffer, parsed-buffer y encoded-buffer deben ser >= 1")
	}
	if cfg.FitWorkers < 1 {
		return fmt.Errorf("fit-workers debe ser >= 1")
	}
	if cfg.Solver != "svd" && cfg.Solver != "normal" && cfg.Solver != "ridge" {
		return fmt.Errorf("solver invalido %q: use svd, normal o ridge", cfg.Solver)
	}
	if cfg.RidgeLambda < 0 {
		return fmt.Errorf("ridge-lambda debe ser >= 0")
	}

	totalStart := profileStart(cfg, "total")

	stageStart := profileStart(cfg, "load_csv_pipeline")
	rows, loaded, skipped, err := loadCSV(cfg)
	profileEnd(cfg, "load_csv_pipeline", stageStart)
	if err != nil {
		return fmt.Errorf("error cargando CSV: %w", err)
	}
	if loaded == 0 {
		return fmt.Errorf("no se cargaron filas validas")
	}

	stageStart = profileStart(cfg, "split_train_test")
	trainRows, testRows, err := splitRows(rows, cfg.TrainYearEnd)
	profileEnd(cfg, "split_train_test", stageStart)
	if err != nil {
		return fmt.Errorf("error separando train/test: %w", err)
	}

	stageStart = profileStart(cfg, "build_encoding_plan")
	plan, err := buildCatEncodingPlan(trainRows, cfg.CatFeatures, cfg.OrdFeatures, cfg.MaxCatLevels)
	profileEnd(cfg, "build_encoding_plan", stageStart)
	if err != nil {
		return fmt.Errorf("error construyendo encoding categorico: %w", err)
	}

	plan.FeatureNames = append(append([]string(nil), cfg.NumericFeatures...), plan.FeatureNames...)

	stageStart = profileStart(cfg, "build_train_matrix")
	trainX := buildDesignMatrix(trainRows, plan)
	trainY := buildTargetVector(trainRows)
	profileEnd(cfg, "build_train_matrix", stageStart)

	stageStart = profileStart(cfg, "build_test_matrix")
	testX := buildDesignMatrix(testRows, plan)
	testY := buildTargetVector(testRows)
	profileEnd(cfg, "build_test_matrix", stageStart)

	featureNames := append([]string(nil), plan.FeatureNames...)
	stageStart = profileStart(cfg, "standardize")
	trainXStd, testXStd, _, stds := standardize(trainX, testX)
	profileEnd(cfg, "standardize", stageStart)

	fmt.Println("============================================================")
	fmt.Println("Entrenamiento regresion lineal (Go + gonum)")
	fmt.Println("============================================================")
	fmt.Printf("Input: %s\n", cfg.InputPath)
	fmt.Printf("Target: %s\n", cfg.TargetCol)
	fmt.Printf("Filas cargadas: %d | Filas omitidas: %d\n", loaded, skipped)
	fmt.Printf("Train rows: %d | Test rows: %d\n", len(trainRows), len(testRows))
	fmt.Printf("Features numericas (%d): %s\n", len(cfg.NumericFeatures), safeJoin(cfg.NumericFeatures))
	fmt.Printf("Categoricas nominales usadas (%d): %s\n", len(plan.NominalColsUsed), safeJoin(plan.NominalColsUsed))
	fmt.Printf("Categoricas ordinales usadas (%d): %s\n", len(plan.OrdinalColsUsed), safeJoin(plan.OrdinalColsUsed))
	if len(plan.DroppedCols) > 0 {
		fmt.Printf("Categoricas omitidas por cardinalidad (%d): %s\n", len(plan.DroppedCols), strings.Join(plan.DroppedCols, ", "))
	}
	fmt.Printf("Total features finales tras encoding: %d\n", len(featureNames))
	fmt.Printf("Train-year-end: %d\n", cfg.TrainYearEnd)
	fmt.Printf(
		"Pipeline concurrente -> parser-workers=%d, encoder-workers=%d, buffers(raw=%d, parsed=%d, encoded=%d)\n",
		cfg.ParserWorkers, cfg.EncoderWorkers, cfg.RawBuffer, cfg.ParsedBuffer, cfg.EncodedBuffer,
	)
	fmt.Printf("Solver -> %s | fit-workers=%d | ridge-lambda=%g\n", cfg.Solver, cfg.FitWorkers, cfg.RidgeLambda)

	if cfg.UsePCA {
		var explained float64
		var k int
		stageStart = profileStart(cfg, "pca")
		trainXStd, testXStd, k, explained, err = applyPCA(trainXStd, testXStd, cfg.PCAVarianceGoal)
		profileEnd(cfg, "pca", stageStart)
		if err != nil {
			return fmt.Errorf("error en PCA: %w", err)
		}
		fmt.Printf("PCA: activado | componentes=%d | varianza explicada=%.4f\n", k, explained)
	} else {
		fmt.Println("PCA: desactivado")
	}

	stageStart = profileStart(cfg, "add_intercept")
	trainXInt := addIntercept(trainXStd)
	testXInt := addIntercept(testXStd)
	profileEnd(cfg, "add_intercept", stageStart)

	stageStart = profileStart(cfg, "fit_linear_regression")
	beta, err := fitLinearRegression(trainXInt, trainY, cfg)
	profileEnd(cfg, "fit_linear_regression", stageStart)
	if err != nil {
		return fmt.Errorf("error entrenando regresion lineal: %w", err)
	}

	stageStart = profileStart(cfg, "predict")
	trainPred := predict(trainXInt, beta)
	testPred := predict(testXInt, beta)
	profileEnd(cfg, "predict", stageStart)

	stageStart = profileStart(cfg, "compute_metrics")
	trainMetrics := computeMetrics(trainY, trainPred)
	testMetrics := computeMetrics(testY, testPred)
	profileEnd(cfg, "compute_metrics", stageStart)

	fmt.Println("\n--- Metricas ---")
	fmt.Printf("Train -> MAE: %.6f | RMSE: %.6f | R2: %.6f\n", trainMetrics.MAE, trainMetrics.RMSE, trainMetrics.R2)
	fmt.Printf("Test  -> MAE: %.6f | RMSE: %.6f | R2: %.6f\n", testMetrics.MAE, testMetrics.RMSE, testMetrics.R2)

	fmt.Println("\n--- Coeficientes (sin intercepto) ---")
	if !cfg.UsePCA {
		printTopCoefficients(beta, featureNames, 20, stds)
	} else {
		printTopPcaCoefficients(beta, 20)
	}
	fmt.Println("============================================================")
	profileEnd(cfg, "total", totalStart)
	return nil
}
