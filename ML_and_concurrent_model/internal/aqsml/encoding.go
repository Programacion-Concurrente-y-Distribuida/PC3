package aqsml

import (
	"fmt"
	"sort"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func buildCatEncodingPlan(trainRows []rowData, catFeatures, ordFeatures []string, maxCatLevels int) (*catEncodingPlan, error) {
	ordSet := make(map[string]struct{}, len(ordFeatures))
	for _, c := range ordFeatures {
		ordSet[c] = struct{}{}
	}

	plan := &catEncodingPlan{
		Nominal:      make([]nominalEncoder, 0),
		Ordinal:      make([]ordinalEncoder, 0),
		DroppedCols:  make([]string, 0),
		FeatureNames: make([]string, 0),
	}

	for colIdx, colName := range catFeatures {
		uniq := make(map[string]struct{})
		for _, rw := range trainRows {
			uniq[rw.CatValues[colIdx]] = struct{}{}
		}
		levels := make([]string, 0, len(uniq))
		for k := range uniq {
			levels = append(levels, k)
		}
		sort.Strings(levels)

		if len(levels) < 2 {
			plan.DroppedCols = append(plan.DroppedCols, fmt.Sprintf("%s(unico_nivel)", colName))
			continue
		}
		if len(levels) > maxCatLevels {
			plan.DroppedCols = append(plan.DroppedCols, fmt.Sprintf("%s(%d_niveles)", colName, len(levels)))
			continue
		}

		_, isOrdinal := ordSet[colName]
		if isOrdinal {
			levelRank := make(map[string]float64, len(levels))
			for i, lv := range levels {
				levelRank[lv] = float64(i)
			}
			plan.Ordinal = append(plan.Ordinal, ordinalEncoder{
				ColName:   colName,
				ColIdx:    colIdx,
				LevelRank: levelRank,
			})
			plan.FeatureNames = append(plan.FeatureNames, colName+"__ord")
			plan.OrdinalColsUsed = append(plan.OrdinalColsUsed, colName)
			continue
		}

		base := levels[0]
		active := levels[1:]
		lvlToOffset := make(map[string]int, len(active))
		for i, lv := range active {
			lvlToOffset[lv] = i
		}
		plan.Nominal = append(plan.Nominal, nominalEncoder{
			ColName:     colName,
			ColIdx:      colIdx,
			BaseLevel:   base,
			ActiveLvls:  active,
			LvlToOffset: lvlToOffset,
		})
		for _, lv := range active {
			plan.FeatureNames = append(plan.FeatureNames, fmt.Sprintf("%s__%s", colName, sanitizeLevel(lv)))
		}
		plan.NominalColsUsed = append(plan.NominalColsUsed, colName)
	}
	return plan, nil
}

func sanitizeLevel(s string) string {
	replacer := strings.NewReplacer(" ", "_", "/", "_", "-", "_", ".", "_", ",", "_", "(", "", ")", "", ":", "_")
	out := replacer.Replace(s)
	if out == "" {
		return "EMPTY"
	}
	return out
}

func buildDesignMatrix(rows []rowData, plan *catEncodingPlan) *mat.Dense {
	if len(rows) == 0 {
		return mat.NewDense(0, 0, nil)
	}

	numCount := len(rows[0].NumValues)
	catCount := len(plan.FeatureNames)
	totalCols := numCount + catCount
	m := mat.NewDense(len(rows), totalCols, nil)

	for i, rw := range rows {
		for j, v := range rw.NumValues {
			m.Set(i, j, v)
		}

		offset := numCount
		for _, enc := range plan.Nominal {
			val := rw.CatValues[enc.ColIdx]
			if pos, ok := enc.LvlToOffset[val]; ok {
				m.Set(i, offset+pos, 1.0)
			}
			offset += len(enc.ActiveLvls)
		}
		for _, enc := range plan.Ordinal {
			val := rw.CatValues[enc.ColIdx]
			if rank, ok := enc.LevelRank[val]; ok {
				m.Set(i, offset, rank)
			} else {
				m.Set(i, offset, 0.0)
			}
			offset++
		}
	}
	return m
}
