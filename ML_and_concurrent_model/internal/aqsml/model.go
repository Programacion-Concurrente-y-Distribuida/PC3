package aqsml

import (
	"fmt"
	"math"
	"sort"
	"sync"

	"gonum.org/v1/gonum/mat"
)

func standardize(trainX, testX *mat.Dense) (*mat.Dense, *mat.Dense, []float64, []float64) {
	rTrain, c := trainX.Dims()
	rTest, _ := testX.Dims()

	means := make([]float64, c)
	stds := make([]float64, c)
	trainOut := mat.NewDense(rTrain, c, nil)
	testOut := mat.NewDense(rTest, c, nil)

	for j := 0; j < c; j++ {
		var sum float64
		for i := 0; i < rTrain; i++ {
			sum += trainX.At(i, j)
		}
		mean := sum / float64(rTrain)
		means[j] = mean

		var sq float64
		for i := 0; i < rTrain; i++ {
			d := trainX.At(i, j) - mean
			sq += d * d
		}
		std := math.Sqrt(sq / float64(rTrain))
		if std == 0 {
			std = 1.0
		}
		stds[j] = std

		for i := 0; i < rTrain; i++ {
			trainOut.Set(i, j, (trainX.At(i, j)-mean)/std)
		}
		for i := 0; i < rTest; i++ {
			testOut.Set(i, j, (testX.At(i, j)-mean)/std)
		}
	}
	return trainOut, testOut, means, stds
}

func applyPCA(trainX, testX *mat.Dense, varianceGoal float64) (*mat.Dense, *mat.Dense, int, float64, error) {
	n, p := trainX.Dims()
	if n <= 1 {
		return nil, nil, 0, 0, fmt.Errorf("filas insuficientes para PCA")
	}

	var svd mat.SVD
	if ok := svd.Factorize(trainX, mat.SVDThin); !ok {
		return nil, nil, 0, 0, fmt.Errorf("no se pudo factorizar SVD")
	}
	vals := svd.Values(nil)
	if len(vals) == 0 {
		return nil, nil, 0, 0, fmt.Errorf("SVD sin valores singulares")
	}

	totalVar := 0.0
	variances := make([]float64, len(vals))
	for i, s := range vals {
		v := (s * s) / float64(n-1)
		variances[i] = v
		totalVar += v
	}

	cum := 0.0
	k := 0
	for i, v := range variances {
		cum += v
		k = i + 1
		if cum/totalVar >= varianceGoal {
			break
		}
	}
	if k > p {
		k = p
	}

	var vmat mat.Dense
	svd.VTo(&vmat)
	components := vmat.Slice(0, p, 0, k)

	trainProj := mat.NewDense(n, k, nil)
	trainProj.Mul(trainX, components)

	nTest, _ := testX.Dims()
	testProj := mat.NewDense(nTest, k, nil)
	testProj.Mul(testX, components)

	return trainProj, testProj, k, cum / totalVar, nil
}

func addIntercept(x *mat.Dense) *mat.Dense {
	r, c := x.Dims()
	out := mat.NewDense(r, c+1, nil)
	for i := 0; i < r; i++ {
		out.Set(i, 0, 1.0)
		for j := 0; j < c; j++ {
			out.Set(i, j+1, x.At(i, j))
		}
	}
	return out
}

func fitLinearRegression(x *mat.Dense, y *mat.Dense, cfg Config) (*mat.Dense, error) {
	switch cfg.Solver {
	case "normal":
		return fitNormalEquationsConcurrent(x, y, cfg.FitWorkers, 0)
	case "ridge":
		return fitNormalEquationsConcurrent(x, y, cfg.FitWorkers, cfg.RidgeLambda)
	case "svd":
		return fitLinearRegressionSVD(x, y)
	default:
		return nil, fmt.Errorf("solver no soportado: %s", cfg.Solver)
	}
}

func fitNormalEquationsConcurrent(x *mat.Dense, y *mat.Dense, workers int, lambda float64) (*mat.Dense, error) {
	n, c := x.Dims()
	if n == 0 {
		return nil, fmt.Errorf("sin filas en entrenamiento")
	}
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}

	type partial struct {
		xtx []float64
		xty []float64
	}

	partials := make([]partial, workers)
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers

	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > n {
			end = n
		}
		if start >= end {
			partials[w] = partial{
				xtx: make([]float64, c*c),
				xty: make([]float64, c),
			}
			continue
		}

		wg.Add(1)
		go func(workerIdx, rowStart, rowEnd int) {
			defer wg.Done()
			localXTX := make([]float64, c*c)
			localXTy := make([]float64, c)

			rowVals := make([]float64, c)
			for i := rowStart; i < rowEnd; i++ {
				for j := 0; j < c; j++ {
					rowVals[j] = x.At(i, j)
				}
				yi := y.At(i, 0)

				for j := 0; j < c; j++ {
					xj := rowVals[j]
					localXTy[j] += xj * yi
					base := j * c
					for k := 0; k <= j; k++ {
						localXTX[base+k] += xj * rowVals[k]
					}
				}
			}

			partials[workerIdx] = partial{
				xtx: localXTX,
				xty: localXTy,
			}
		}(w, start, end)
	}
	wg.Wait()

	xtxData := make([]float64, c*c)
	xtyData := make([]float64, c)
	for _, p := range partials {
		for i := 0; i < c*c; i++ {
			xtxData[i] += p.xtx[i]
		}
		for i := 0; i < c; i++ {
			xtyData[i] += p.xty[i]
		}
	}

	for j := 0; j < c; j++ {
		for k := 0; k < j; k++ {
			xtxData[k*c+j] = xtxData[j*c+k]
		}
	}

	if lambda > 0 {
		// No regularizamos el intercepto (columna 0).
		for j := 1; j < c; j++ {
			xtxData[j*c+j] += lambda
		}
	}

	xtx := mat.NewSymDense(c, xtxData)
	xty := mat.NewDense(c, 1, xtyData)
	beta := mat.NewDense(c, 1, nil)

	var chol mat.Cholesky
	if ok := chol.Factorize(xtx); ok {
		if err := chol.SolveTo(beta, xty); err == nil {
			return beta, nil
		}
	}

	// Fallback robusto si XTX es singular o casi singular.
	return fitLinearRegressionSVD(x, y)
}

func fitLinearRegressionSVD(x *mat.Dense, y *mat.Dense) (*mat.Dense, error) {
	n, c := x.Dims()
	if n == 0 {
		return nil, fmt.Errorf("sin filas en entrenamiento")
	}
	beta := mat.NewDense(c, 1, nil)

	var svd mat.SVD
	if ok := svd.Factorize(x, mat.SVDThin); !ok {
		return nil, fmt.Errorf("fallo en factorization SVD")
	}
	singularValues := svd.Values(nil)
	if len(singularValues) == 0 {
		return nil, fmt.Errorf("SVD devolvio 0 valores singulares")
	}

	var u mat.Dense
	svd.UTo(&u)
	var v mat.Dense
	svd.VTo(&v)

	utY := mat.NewDense(c, 1, nil)
	utY.Mul(u.T(), y)

	maxS := singularValues[0]
	tol := 1e-12 * maxS
	for i, s := range singularValues {
		if s > tol {
			utY.Set(i, 0, utY.At(i, 0)/s)
		} else {
			utY.Set(i, 0, 0)
		}
	}

	beta.Mul(&v, utY)
	return beta, nil
}

func predict(x *mat.Dense, beta *mat.Dense) *mat.Dense {
	var out mat.Dense
	out.Mul(x, beta)
	return &out
}

func computeMetrics(yTrue, yPred *mat.Dense) metrics {
	r, _ := yTrue.Dims()
	if r == 0 {
		return metrics{}
	}
	var sumAbs, sumSq, sumTrue float64
	for i := 0; i < r; i++ {
		sumTrue += yTrue.At(i, 0)
	}
	meanTrue := sumTrue / float64(r)

	var ssTot float64
	for i := 0; i < r; i++ {
		t := yTrue.At(i, 0)
		p := yPred.At(i, 0)
		err := t - p
		sumAbs += math.Abs(err)
		sumSq += err * err
		d := t - meanTrue
		ssTot += d * d
	}

	r2 := 1.0
	if ssTot > 0 {
		r2 = 1 - (sumSq / ssTot)
	}
	return metrics{
		MAE:  sumAbs / float64(r),
		RMSE: math.Sqrt(sumSq / float64(r)),
		R2:   r2,
	}
}

func printTopCoefficients(beta *mat.Dense, features []string, topN int, stds []float64) {
	type coef struct {
		Name    string
		Value   float64
		Abs     float64
		OrigEst float64
	}
	coefs := make([]coef, 0, len(features))
	for j, name := range features {
		val := beta.At(j+1, 0)
		orig := val / stds[j]
		coefs = append(coefs, coef{
			Name:    name,
			Value:   val,
			Abs:     math.Abs(val),
			OrigEst: orig,
		})
	}
	sort.Slice(coefs, func(i, j int) bool { return coefs[i].Abs > coefs[j].Abs })
	if topN > len(coefs) {
		topN = len(coefs)
	}
	fmt.Printf("Intercepto: %.6f\n", beta.At(0, 0))
	for i := 0; i < topN; i++ {
		c := coefs[i]
		fmt.Printf("%2d) %-40s coef_z=%.6f | coef_aprox_escala_original=%.8f\n", i+1, c.Name, c.Value, c.OrigEst)
	}
}

func printTopPcaCoefficients(beta *mat.Dense, topN int) {
	type coef struct {
		Name  string
		Value float64
		Abs   float64
	}
	_, c := beta.Dims()
	coefs := make([]coef, 0, c-1)
	for j := 1; j < c; j++ {
		v := beta.At(j, 0)
		coefs = append(coefs, coef{
			Name:  fmt.Sprintf("PC%d", j),
			Value: v,
			Abs:   math.Abs(v),
		})
	}
	sort.Slice(coefs, func(i, j int) bool { return coefs[i].Abs > coefs[j].Abs })
	if topN > len(coefs) {
		topN = len(coefs)
	}
	fmt.Printf("Intercepto: %.6f\n", beta.At(0, 0))
	for i := 0; i < topN; i++ {
		cf := coefs[i]
		fmt.Printf("%2d) %-8s coef=%.6f\n", i+1, cf.Name, cf.Value)
	}
}
