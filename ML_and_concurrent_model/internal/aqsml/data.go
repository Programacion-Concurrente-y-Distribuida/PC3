package aqsml

import (
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"gonum.org/v1/gonum/mat"
)

type csvLayout struct {
	yearIdx   int
	targetIdx int
	numIdxs   []int
	catIdxs   []int
}

type rawCSVRow struct {
	record []string
}

type parsedRow struct {
	year      int
	numValues []float64
	catRaw    []string
	target    float64
}

func loadCSV(cfg Config) ([]rowData, int, int, error) {
	f, err := os.Open(cfg.InputPath)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.ReuseRecord = true

	header, err := reader.Read()
	if err != nil {
		return nil, 0, 0, err
	}

	layout, err := buildCSVLayout(header, cfg.NumericFeatures, cfg.CatFeatures, cfg.TargetCol)
	if err != nil {
		return nil, 0, 0, err
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	rawRowsCh := make(chan rawCSVRow, cfg.RawBuffer)
	parsedCh := make(chan parsedRow, cfg.ParsedBuffer)
	encodedCh := make(chan rowData, cfg.EncodedBuffer)

	var loaded atomic.Int64
	var skipped atomic.Int64

	// Reader
	go func() {
		defer close(rawRowsCh)
		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			record, e := reader.Read()
			if errors.Is(e, io.EOF) {
				return
			}
			if e != nil {
				skipped.Add(1)
				continue
			}

			// Copy is mandatory because csv.Reader may reuse backing array.
			cp := append([]string(nil), record...)
			select {
			case <-ctx.Done():
				return
			case rawRowsCh <- rawCSVRow{record: cp}:
			}
		}
	}()

	// Parser pool
	var parserWG sync.WaitGroup
	for i := 0; i < cfg.ParserWorkers; i++ {
		parserWG.Add(1)
		go func() {
			defer parserWG.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case raw, ok := <-rawRowsCh:
					if !ok {
						return
					}
					pr, ok := parseRawRow(raw, layout)
					if !ok {
						skipped.Add(1)
						continue
					}
					select {
					case <-ctx.Done():
						return
					case parsedCh <- pr:
					}
				}
			}
		}()
	}
	go func() {
		parserWG.Wait()
		close(parsedCh)
	}()

	// Encoder pool
	var encoderWG sync.WaitGroup
	for i := 0; i < cfg.EncoderWorkers; i++ {
		encoderWG.Add(1)
		go func() {
			defer encoderWG.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case pr, ok := <-parsedCh:
					if !ok {
						return
					}
					encoded, ok := encodeParsedRow(pr)
					if !ok {
						skipped.Add(1)
						continue
					}

					// Respect max valid rows if requested.
					newCount := loaded.Add(1)
					if cfg.MaxRows > 0 && int(newCount) > cfg.MaxRows {
						loaded.Add(-1)
						cancel()
						return
					}

					select {
					case <-ctx.Done():
						return
					case encodedCh <- encoded:
					}
				}
			}
		}()
	}
	go func() {
		encoderWG.Wait()
		close(encodedCh)
	}()

	out := make([]rowData, 0, 32768)
	for row := range encodedCh {
		out = append(out, row)
	}

	if len(out) == 0 {
		return out, int(loaded.Load()), int(skipped.Load()), fmt.Errorf("no hubo filas validas tras el pipeline")
	}
	return out, int(loaded.Load()), int(skipped.Load()), nil
}

func buildCSVLayout(header []string, numFeatures, catFeatures []string, target string) (csvLayout, error) {
	colIdx := make(map[string]int, len(header))
	for i, h := range header {
		colIdx[strings.TrimSpace(h)] = i
	}

	required := append([]string{"Year", target}, numFeatures...)
	required = append(required, catFeatures...)
	missingCols := make([]string, 0)
	for _, c := range required {
		if _, ok := colIdx[c]; !ok {
			missingCols = append(missingCols, c)
		}
	}
	if len(missingCols) > 0 {
		sort.Strings(missingCols)
		return csvLayout{}, fmt.Errorf("columnas no encontradas: %s", strings.Join(missingCols, ", "))
	}

	numIdxs := make([]int, len(numFeatures))
	for i, c := range numFeatures {
		numIdxs[i] = colIdx[c]
	}
	catIdxs := make([]int, len(catFeatures))
	for i, c := range catFeatures {
		catIdxs[i] = colIdx[c]
	}

	return csvLayout{
		yearIdx:   colIdx["Year"],
		targetIdx: colIdx[target],
		numIdxs:   numIdxs,
		catIdxs:   catIdxs,
	}, nil
}

func parseRawRow(raw rawCSVRow, layout csvLayout) (parsedRow, bool) {
	record := raw.record

	year, ok := parseInt(record[layout.yearIdx])
	if !ok {
		return parsedRow{}, false
	}
	targetVal, ok := parseFloat(record[layout.targetIdx])
	if !ok {
		return parsedRow{}, false
	}

	numVals := make([]float64, len(layout.numIdxs))
	for i, idx := range layout.numIdxs {
		v, ok := parseFloat(record[idx])
		if !ok {
			return parsedRow{}, false
		}
		numVals[i] = v
	}

	catRaw := make([]string, len(layout.catIdxs))
	for i, idx := range layout.catIdxs {
		catRaw[i] = record[idx]
	}

	return parsedRow{
		year:      year,
		numValues: numVals,
		catRaw:    catRaw,
		target:    targetVal,
	}, true
}

func encodeParsedRow(pr parsedRow) (rowData, bool) {
	catVals := make([]string, len(pr.catRaw))
	for i, raw := range pr.catRaw {
		v := strings.TrimSpace(raw)
		if v == "" {
			return rowData{}, false
		}
		catVals[i] = v
	}
	return rowData{
		Year:      pr.year,
		NumValues: pr.numValues,
		CatValues: catVals,
		Target:    pr.target,
	}, true
}

func splitRows(rows []rowData, trainYearEnd int) ([]rowData, []rowData, error) {
	trainRows := make([]rowData, 0, len(rows))
	testRows := make([]rowData, 0, len(rows)/5)
	for _, rw := range rows {
		if rw.Year <= trainYearEnd {
			trainRows = append(trainRows, rw)
		} else {
			testRows = append(testRows, rw)
		}
	}
	if len(trainRows) == 0 || len(testRows) == 0 {
		return nil, nil, fmt.Errorf("split vacio (train=%d, test=%d)", len(trainRows), len(testRows))
	}
	return trainRows, testRows, nil
}

func buildTargetVector(rows []rowData) *mat.Dense {
	y := mat.NewDense(len(rows), 1, nil)
	for i, rw := range rows {
		y.Set(i, 0, rw.Target)
	}
	return y
}

func parseFloat(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, false
	}
	return v, true
}

func parseInt(s string) (int, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, false
	}
	return int(v), true
}
